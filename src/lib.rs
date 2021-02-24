#![allow(clippy::upper_case_acronyms)]

//! A simple crate implementing a struct wrapping a reader and writer that's used to create readers
//! to unique data.

pub(crate) use ::std::{
    cmp::Ordering::{self, Equal},
    hash::{Hash, Hasher},
    io::{self, prelude::*, Cursor, SeekFrom},
    ptr,
    slice,
    str,
    sync::{Mutex, PoisonError},
};

#[cfg(feature = "serde_support")]
pub(crate) use ::{
    serde::{
        de::{Deserialize, Deserializer},
        ser::{Serialize, SerializeSeq, SerializeStruct, Serializer},
    },
    serde_derive::*,
    std::{
        mem,
        sync::RwLock,
    },
};

/// A struct wrapping a `Mutex<T>` used for storing and retrieving data thought readers.
///
/// Note that `T` it's wrapped into a [`Mutex`] for ensure [`IOObj`] does not lock access to the
/// `IOInterner` and to guarantee will only lock at `Read` methods.
pub struct IOInterner<T: Write + Read + Seek + ?Sized> {
    /// The underlying writer wrapped into a `Mutex`.
    ///
    /// The data pointed by existing [`IOEntry`] instances would change and thus the comparison
    /// algorithm would mess up if you do not ensure that already existing data remain equal after
    /// releasing a lock,it's advisable to only write at the end.
    pub inner: Mutex<T>,
}

impl<T: Write + Read + Seek> IOInterner<T> {
    /// Create a new `IOInterner` that uses as common storage `x`.
    #[inline]
    pub fn new(x: T) -> Self {
        Self {
            inner: Mutex::new(x),
        }
    }

    /// Create a new `IOInterner` that uses as common storage `x`.
    #[inline]
    pub fn from_buf<U>(x: U) -> IOInterner<Cursor<U>> where Cursor<U>: Write + Read + Seek {
        IOInterner::new(Cursor::new(x))
    }

    /// Returns an `IOInterner<U>` after applying the closure `f` to it's inner value.
    /// 
    /// # Errors
    /// 
    /// See [`Mutex::into_inner`].
    pub fn map<U: Write + Read + Seek, F: FnOnce(T) -> U>(self, f: F) -> Result<IOInterner<U>, PoisonError<T>> {
        self.inner.into_inner().map(|e| IOInterner::new(f(e)))
    }
}

impl<T: Write + Read + Seek + ?Sized> IOInterner<T> {
    /// Convenience for `Self::new(Cursor::new(x))`.
    #[inline]
    pub fn from_vec(x: Vec<u8>) -> IOInterner<Cursor<Vec<u8>>> {
        IOInterner::new(Cursor::new(x))
    }

    /// Invokes [`Write::flush`] on `self.inner`.
    ///
    /// # Panics
    ///
    /// This function panics if the [`Mutex`] it's poisoned.
    #[inline]
    pub fn flush(&self) -> io::Result<()> {
        self.inner.lock().unwrap().flush()
    }

    /// Like [`Self::get_or_intern`] instead that will return also a `bool` indicating whether the
    /// entry was not there before.
    /// 
    /// # Errors
    /// 
    /// See [`Self::get_or_intern`].
    pub fn try_intern<U: Read + Seek>(&self, mut buf: U) -> io::Result<(IOEntry<'_, T>, bool)> {
        let mut l = self.inner.lock().unwrap();

        let buf_len = buf.seek(SeekFrom::End(0))?;

        if buf_len == 0 {
            return Ok((IOEntry {
                start_init: 0,
                len: 0,
                guard: &self.inner,
            }, false));
        }

        let len = l.seek(SeekFrom::End(0))?;

        let mut pos = IOPos {
            start_pos: len,
            len: buf_len,
        };

        for start in 0..len.saturating_sub(buf_len) {
            l.seek(SeekFrom::Start(start))?;
            buf.seek(SeekFrom::Start(0))?;

            if starts_with(&mut *l, &mut buf)? {
                pos.start_pos = start;
                return Ok((self.get_pos(pos), false));
            }
        }

        l.seek(SeekFrom::Start(len))?;
        buf.seek(SeekFrom::Start(0))?;
        io::copy(&mut buf, &mut *l)?;
        l.flush()?;

        Ok((self.get_pos(pos), true))
    }

    /// Creates a new `IOEntry` object that will be able to generate [`IOObj`] that always read
    /// same bytes as `buf` from `T` so `buf` will be written at the end if it's not already.
    ///
    /// The [`IOObj`] does not implement `Write` as that would mess up equality of two
    /// [`IOEntry`] instances pointing to the same position with same length which it's actually
    /// the comparison algorithm.
    /// 
    /// # Errors
    /// 
    /// This function returns any [`io::Error`] that could result from [`seek`] and [`read`]
    /// operation applied to `buf` or any [`seek`],[`read`] or [`write`] one applied to
    /// `self.inner`.
    ///
    /// # Panics
    ///
    /// This function panics if the [`Mutex`] it's poisoned.
    /// 
    /// [`seek`][`Seek::seek`]
    /// [`write`][`Write::write`]
    /// [`read`][`Read::read`]
    pub fn get_or_intern<U: Read + Seek>(&self, buf: U) -> io::Result<IOEntry<'_, T>> {
        self.try_intern(buf).map(|e| e.0)
    }

    /// Convenience for `self.get_or_intern(io::Cursor::new(bytes))`.
    pub fn get_or_intern_bytes(&self, bytes: impl AsRef<[u8]>) -> io::Result<IOEntry<'_, T>> {
        self.get_or_intern(Cursor::new(bytes))
    }

    /// Convenience for `self.try_intern(io::Cursor::new(bytes))`.
    pub fn try_intern_bytes(&self, bytes: impl AsRef<[u8]>) -> io::Result<(IOEntry<'_, T>, bool)> {
        self.try_intern(Cursor::new(bytes))
    }

    /// Creates an [`IOEntry`] out it's first [`IOPos`].
    #[inline]
    pub fn get_pos(&self, pos: IOPos) -> IOEntry<'_, T> {
        IOEntry {
            start_init: pos.start_pos,
            len: pos.len,
            guard: &self.inner,
        }
    }
}

/// A struct generated by [`IOInterner::get_or_intern`] or [`IOInterner::try_intern`].
pub struct IOEntry<'a, T: ?Sized> {
    start_init: u64,
    len: u64,
    guard: &'a Mutex<T>,
}

impl<'a, T: ?Sized> Clone for IOEntry<'a, T> {
    fn clone(&self) -> Self {
        unsafe { ptr::read(self) }
    }
}

impl<'a, T: ?Sized> IOEntry<'a, T> {
    /// Creates a new [`IOObj`] that can be [`Read`] and [`Seek`].
    pub fn get_object(&self) -> IOObj<'a, T> {
        IOObj {
            start_init: self.start_init,
            start: self.start_init,
            len: self.len,
            guard: self.guard,
        }
    }

    /// Convenience for `self.get_object().position()`,see [`IOObj::position`].
    #[inline]
    pub fn position(&self) -> IOPos {
        self.get_object().position()
    }
}

impl<'a, T: ?Sized> PartialEq for IOEntry<'a, T> {
    /// Compares the start position of `self` in the interner and it's length to `other`;returning
    /// `false` if either are different,`true` otherwise.
    ///
    /// # Panics
    ///
    /// It panics if `self` and `other` were created from different [`IOInterner`]s.
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.guard as *const _, other.guard as *const _, "entries from different interners cannot be compared,consider use `IOEntry::get_object` to create `IOObj`ects");

        self.position() == other.position()
    }
}

impl<'a, T: ?Sized> Eq for IOEntry<'a, T> {}

impl<'a, T: ?Sized> Hash for IOEntry<'a, T> {
    /// Hash the start position of the entry and the length.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.position().hash(state)
    }
}

/// A struct generated by [`IOEntry::get_object`].
pub struct IOObj<'a, T: ?Sized> {
    start_init: u64,
    start: u64,
    len: u64,
    guard: &'a Mutex<T>,
}

impl<'a, T: ?Sized> Clone for IOObj<'a, T> {
    fn clone(&self) -> Self {
        unsafe { ptr::read(self) }
    }
}

impl<'a, T: ?Sized> IOObj<'a, T> {
    /// Converts back to an [`IOEntry`].
    #[inline]
    pub fn to_entry(&self) -> IOEntry<'a, T> {
        let mut a = self.clone();
        a.seek(SeekFrom::Start(0)).unwrap();

        IOEntry {
            start_init: a.start,
            len: a.len,
            guard: a.guard,
        }
    }

    fn fields_eq(&self, other: &Self) -> bool {
        ptr::eq(self.guard, other.guard) && self.position() == other.position()
    }

    /// See [`IOPos`].
    pub fn position(&self) -> IOPos {
        IOPos {
            start_pos: self.start,
            len: self.len,
        }
    }
}

impl<'a, T: Read + Seek + ?Sized> Read for IOObj<'a, T> {
    /// Invokes `Read::read` in the underlying reader seeking to the position this entry starts
    /// and only taking as much bytes as the length of this entry.
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.len == 0 {
            return Ok(0);
        }

        let mut l = self.guard.lock().unwrap();
        l.seek(SeekFrom::Start(self.start))?;

        let len = Read::take(&mut *l, self.len).read(buf)?;
        self.seek(SeekFrom::Current(len as _))?;
        Ok(len)
    }
}

impl<'a, T: Read + Seek + ?Sized> PartialEq for IOObj<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.fields_eq(other)
            || eq(self.clone(), other.clone()).expect("io error while testing for equality")
    }
}

impl<'a, T: Read + Seek + ?Sized> Eq for IOObj<'a, T> {}

impl<'a, T: Read + Seek + ?Sized> Hash for IOObj<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash(self.clone(), state).expect("io error while hashing")
    }
}

impl<'a, T: Read + Seek + ?Sized> PartialOrd for IOObj<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, T: Read + Seek + ?Sized> Ord for IOObj<'a, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.fields_eq(other) {
            return Equal;
        }

        cmp(self.clone(), other.clone()).expect("io error while comparing the order")
    }
}

impl<'a, T: ?Sized> Seek for IOObj<'a, T> {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Current(x) => {
                if x.is_negative() {
                    let x = x.abs() as u64;

                    let new_start = self.start - x;

                    if new_start < self.start_init {
                        return Err(io::ErrorKind::InvalidInput.into());
                    }

                    self.start = new_start;
                    self.len += x;
                } else {
                    let x = x.abs() as u64;

                    if x > self.len {
                        self.start += self.len;
                        self.len = 0;
                    } else {
                        self.start += x;
                        self.len -= x;
                    }
                }

                Ok(self.start - self.start_init)
            }
            SeekFrom::Start(x) => {
                let new_start = self.start_init + x;

                if new_start > self.start {
                    self.seek(SeekFrom::Current((new_start - self.start) as i64))
                } else {
                    self.seek(SeekFrom::Current(-((self.start - new_start) as i64)))
                }
            }
            SeekFrom::End(x) => {
                self.start += self.len;
                self.len = 0;

                self.seek(SeekFrom::Current(x))
            }
        }
    }
}

/// Struct stating which data [`IOObj`] will read from the [`IOInterner`] internal IO object.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct IOPos {
    /// The start position on which the [`IOObj`] can read in the next [`Read::read`].
    pub start_pos: u64,
    /// The max amount of bytes this [`IOObj`] can read in the next [`Read::read`].
    pub len: u64,
}

impl PartialOrd for IOPos {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IOPos {
    fn cmp(&self, other: &Self) -> Ordering {
        self.len.cmp(&other.len)
    }
}

/// Internal buffer length used in many free functions of this crate,put here if in the future some
/// way of create it at compile time it's implemented,only thinkable way it's
/// `option_env!("IOINTERNER_BUF_LEN").and_then(|e| e.trim().parse().ok()).unwrap_or(512)` but,
/// except for `option_env!` and `Option::{and_then, unwrap_or}`,cannot be put into a constant.
pub const BUF_LEN: usize = 512;

/// Pass to `callback` slices with lenght up to 512 contained in both readers until either one
/// reach EOF and returns `None` or when `callback` returns `Some`,that case this function will
/// return it.
///
/// # Errors
///
/// See [`io::copy`].
pub fn io_op<R1: Read, R2: Read, T>(
    mut x: R1,
    mut y: R2,
    mut callback: impl FnMut(&[u8], &[u8]) -> Option<T>,
) -> io::Result<Option<T>> {
    let mut buf1 = [0; BUF_LEN];
    let mut buf2 = [0; BUF_LEN];

    Ok(loop {
        let mut buf1r = &mut buf1[..];
        let mut buf2r = &mut buf2[..];

        let mut x = Read::take(&mut x, buf1r.len() as _);
        let mut y = Read::take(&mut y, buf1r.len() as _);

        let readed1 = io::copy(&mut x, &mut buf1r)? as usize;
        let readed2 = io::copy(&mut y, &mut buf2r)? as usize;

        let a = callback(&buf1[..readed1], &buf2[..readed2]);

        if a.is_some() {
            break a;
        }

        if readed1 == 0 || readed2 == 0 {
            break None;
        }
    })
}

/// Checks if the contents of the reader `x` are the ones of `y`.
///
/// # Errors
///
/// See [`io_op`].
pub fn eq<R1: Read, R2: Read>(x: R1, y: R2) -> io::Result<bool> {
    io_op(x, y, |x, y| if x == y { None } else { Some(()) }).map(|e| e.is_none())
}

/// Checks if the first contents of the reader `haystack` are the ones of `needle`,an empty needle
/// is always true.
///
/// # Errors
///
/// See [`io_op`].
pub fn starts_with<R1: Read, R2: Read>(haystack: R1, needle: R2) -> io::Result<bool> {
    io_op(haystack, needle, |haystack, needle| {
        if haystack.starts_with(needle) {
            None
        } else {
            Some(())
        }
    }).map(|e| e.is_none())
}

/// Compares the contents of `x` to the ones of `y`,see
/// [`lexicographical comparison`][`Ord#lexicographical-comparison`].
///
/// # Errors
///
/// See [`io_op`].
pub fn cmp<R1: Read, R2: Read>(x: R1, y: R2) -> io::Result<Ordering> {
    io_op(x, y, |x, y| match x.cmp(y) {
        Equal => None,
        x => Some(x),
    })
    .map(|e| e.unwrap_or(Equal))
}

/// Hash the contents of the reader `x` to `state`.
///
/// # Errors
///
/// See [`io_unary_op`].
pub fn hash<R1: Read, H: Hasher>(x: R1, state: &mut H) -> io::Result<()> {
    let mut len = 0;

    let _: Option<()> = io_unary_op(x, |x| {
        Hash::hash_slice(x, state);
        len += x.len();
        None
    })?;

    len.hash(state);

    Ok(())
}

/// Convert the contents of the `reader` into valid UTF-8 and pass it to callback,passing in `None`
/// when some bytes are.
/// 
/// # Errors
/// 
/// See [`io_unary_op`].
pub fn to_utf8<R: Read, F: FnMut(Option<&str>)>(reader: R, mut callback: F) -> io::Result<()> {
        let mut invalid_slice: Option<usize> = None;
        let mut invalid_buf = [0; 4];

        io_unary_op(reader, |mut stream| {
            if let Some(mut len) = invalid_slice.take() {
                let orig_len = len;
                let mut valid = true;

                for b in stream.iter() {
                    len += 1;
                    valid = len <= 4;    

                    if (*b as i8) >= -0x40 {
                        break;                                                    
                    }
                }

                let dlen = len - orig_len;

                let (may_valid, valid_one) = stream.split_at(dlen);

                if valid {
                    // directly after `may_valid` it's `valid_one` so it's impossible to the error
                    // to be expecting bytes that are ahead,the `dlen` byte it's an codepoint
                    // starting one
                    if to_utf8_internal(may_valid, &mut callback).is_some() {
                        callback(None)
                    }
                } else {
                    callback(None);
                }

                stream = valid_one;
            }

            if let Some(e) = to_utf8_internal(stream, &mut callback) {
                let slice = &mut invalid_buf[..e.len()];
                slice.copy_from_slice(e);
                invalid_slice = Some(slice.len());
            }

            None
        }).map(|_: Option<()>| ())
}

fn to_utf8_internal<F: FnMut(Option<&str>)>(mut input: &[u8], mut callback: F) -> Option<&[u8]> {
    loop {
        match str::from_utf8(input) {
            Err(error) => {
                let (valid, after_valid) = input.split_at(error.valid_up_to());

                unsafe {
                   callback(Some(str::from_utf8_unchecked(valid)))
                }

                if let Some(invalid_sequence_length) = error.error_len() {
                    callback(None);
                    input = &after_valid[invalid_sequence_length..]
                } else {
                    break Some(after_valid)
                }
            }
            x => {
                callback(x.ok());
                break None
            }
        }
    }
}

/// Like [`io_op`],but with only one reader and slice passed down to `callback`.
///
/// # Errors
///
/// See [`io::copy`].
pub fn io_unary_op<R1: Read, T>(
    mut x: R1,
    mut callback: impl FnMut(&[u8]) -> Option<T>,
) -> io::Result<Option<T>> {

    let mut buf1 = [0; BUF_LEN];

    Ok(loop {
        let mut buf1r = &mut buf1[..];

        let mut x = Read::take(&mut x, buf1r.len() as _);

        let readed1 = io::copy(&mut x, &mut buf1r)? as usize;

        let a = callback(&buf1[..readed1]);

        if a.is_some() {
            break a;
        }

        if readed1 == 0 {
            break None;
        }
    })
}

#[cfg(feature = "serde_support")]
pub mod serde {
    use super::*;

    #[inline]
    fn stream_len(mut x: impl Seek) -> io::Result<u64> {
        let pos = x.seek(SeekFrom::Current(0))?;
        let len = x.seek(SeekFrom::End(0))?;

        if pos == len {
            return Ok(len)
        }

        x.seek(SeekFrom::Start(pos))?;

        Ok(len)
    }

    impl<T: Write + Read + Seek + Serialize> Serialize for IOInterner<T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.inner.serialize(serializer)
        }
    }

    impl<'a, T: Write + Read + Seek + Deserialize<'a>> Deserialize<'a> for IOInterner<T> {
        fn deserialize<D: Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
            Ok(Self {
                inner: <Mutex<T>>::deserialize(deserializer)?
            })
        }
    }

    /// Serializes the contents of `x` to `serializer`.
    ///
    /// # Errors
    ///
    /// if [`io_op`] fails an `Err` variant of `io::Result` it's returned,otherwise a
    /// `Result<S::Ok, S::Error>`.
    pub fn serialize<R1: Seek + Read, S: Serializer>(
        mut x: R1,
        serializer: S,
    ) -> io::Result<Result<S::Ok, S::Error>> {
        Ok(loop {
            let mut serializer = match serializer.serialize_seq(Some(stream_len(&mut x).map(|e| e as usize / BUF_LEN)? as _)) {
                Ok(e) => e,
                Err(e) => break Err(e),
            };

            break io_unary_op(x, |x| serializer.serialize_element(x).err())?
                .map(|x| Err(x))
                .unwrap_or_else(|| serializer.end());
        })
    }

    /// Error variant of used by the fn [`serialize_flatten`].
    #[derive(Debug)]
    pub enum SerIOErr<S: Serializer> {
        IO(io::Error),
        Ser(S::Error),
    }

    /// Convenience to the fn [`serialize`] except for that it flattens the result thought
    /// returning `Ok` when the former does `Ok(Ok)`,and wraps the two diferrent `Err` variants
    /// into a [`SerIOErr`].
    pub fn serialize_flatten<R1: Seek + Read, S: Serializer>(
        x: R1,
        serializer: S,
    ) -> Result<S::Ok, SerIOErr<S>> {
        Err(match serialize(x, serializer) {
            Ok(Ok(x)) => return Ok(x),
            Ok(Err(e)) => SerIOErr::Ser(e),
            Err(e) => SerIOErr::IO(e),
        })
    }

    /// Struct [io entries][`IOEntry`] serialize to.
    #[derive(Serialize, Deserialize)]
    pub struct ToIOEntry<T: Read + Write + Seek> {
        /// The id of the interner field,other `Self` instances with the same one would be able
        /// to get entries thought [`IOInterner::get_pos`] on the same interner,and exactly one
        /// instance of those will have that interner wrapped into a `Some`.
        pub interner_id: usize,
        /// The interner for the entry serialized,wrapped into an option because more than one
        /// [`IOEntry`] could share it and will be serialized only once.
        pub interner: Option<IOInterner<T>>,
        /// The position of the [`IOEntry`],useful to use [`IOInterner::get_pos`] on the original
        /// interner as described in `interner_id`.
        pub pos: IOPos,
    }

    impl<'a, T: Read + Write + Seek> ToIOEntry<T> {
        #[inline]
        fn fields(&self) -> (IOPos, usize, u8) {
            // the contents of the interner have not to be checked at testing for equality
            // because the algorithm test before `interner_id`,if two `ToIOEntry` share same one
            // both `interner`s cannot be `Some`,so they are told to be equal if both are `None`.
            (self.pos, self.interner_id, self.interner.is_some() as _)
        }
    }

    impl<'a, T: Read + Write + Seek> PartialEq for ToIOEntry<T> {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.fields() == other.fields()
        }
    }

    impl<'a, T: Read + Write + Seek> Eq for ToIOEntry<T> {}

    impl<'a, T: Read + Write + Seek> Hash for ToIOEntry<T> {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.fields().hash(state)
        }
    }

    impl<'a, T: Read + Write + Seek> PartialOrd for ToIOEntry<T> {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<'a, T: Read + Write + Seek> Ord for ToIOEntry<T> {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            self.fields().cmp(&other.fields())
        }
    }

    impl<'a, T: Serialize + Deserialize<'a>> Serialize for IOEntry<'a, T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let transform = |x: usize| -> usize { x.rotate_left(x.leading_zeros()) };

    fn contains(needle: usize, x: &[usize]) -> bool {
        fn bytes<T: ?Sized>(x: &T) -> &[u8] {
            unsafe { slice::from_raw_parts(x as *const _ as *const u8, mem::size_of_val(x)) }
        }

        let needle = bytes(&needle);
        let mut haystack = bytes(&x);

        while let Some(e) = memchr::memchr(needle[0], haystack) {
            haystack = &haystack[e + 1..];

            if haystack.starts_with(&needle[1..]) {
                return true
            }
        }

        false
    }

            let interner_id = transform(self.guard as *const _ as _);

            lazy_static::lazy_static! {
                static ref SERIALIZED: RwLock<Vec<usize>> = RwLock::new(Vec::new());
            }

            let first_entry = contains(interner_id, &**SERIALIZED.read().unwrap());

            if first_entry {
                SERIALIZED.write().unwrap().push(interner_id);
            }

            let mut s = serializer.serialize_struct("ToIOEntry", 3)?;
            s.serialize_field("interner_id", &interner_id)?;
            s.serialize_field("interner", &bool::then(first_entry, || self.guard))?;
            s.serialize_field("pos", &self.position())?;
            s.end()
        }
    }

    impl<'a, T: Seek + Read> Serialize for IOObj<'a, T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serialize(self.clone(), serializer).expect("io error while serializing")
        }
    }
}
