#![allow(clippy::upper_case_acronyms)]

//! A simple crate implementing a struct wrapping a reader and writer that's used to create readers
//! to unique data.

#![allow(unused)]

pub(crate) use ::std::{
    cmp::Ordering::{self, Equal},
    collections::VecDeque,
    convert::TryFrom,
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
        ser::{Serialize, SerializeSeq, Serializer},
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
#[derive(Default, Debug)]
pub struct IOInterner<T: Write + Read + Seek + ?Sized> {
    /// The underlying writer wrapped into a `Mutex`.
    ///
    /// The data pointed by existing [`IOEntry`] instances would change and thus the comparison
    /// algorithm would mess up if you do not ensure that already existing data remain equal after
    /// releasing a lock,it's advisable to only write at the end.
    pub inner: Mutex<T>,
}

impl<T: Write + Read + Seek + Clone> Clone for IOInterner<T> {
    fn clone(&self) -> Self {
        Self::new(Clone::clone(&*self.inner.lock().unwrap_or_else(|e| e.into_inner())))
    }
}

impl<T: Write + Read + Seek + ?Sized> Write for IOInterner<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Write::write(&mut &*self, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Write::flush(&mut &*self)
    }
}

impl<'a, T: Write + Read + Seek + ?Sized> Write for &'a IOInterner<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.lock().unwrap().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.lock().unwrap().flush()
    }
}

impl<T: Write + Read + Seek + ?Sized> Read for IOInterner<T> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        Read::read(&mut &*self, buf)
    }
}

impl<'a, T: Write + Read + Seek + ?Sized> Read for &'a IOInterner<T> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.lock().unwrap().read(buf)
    }
}

impl<T: Write + Read + Seek + ?Sized> Seek for IOInterner<T> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        Seek::seek(&mut &*self, pos)
    }
}

impl<'a, T: Write + Read + Seek + ?Sized> Seek for &'a IOInterner<T> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.lock().unwrap().seek(pos)
    }
}

impl<T: Write + Read + Seek> IOInterner<T> {
    fn from_inner(inner: Mutex<T>) -> Self {
        Self {
            inner
        }
    }

    /// Create a new `IOInterner` that uses as common storage `x`.
    #[inline]
    pub fn new(x: T) -> Self {
        Self::from_inner(Mutex::new(x))
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
        self.inner.into_inner().map(f).map(IOInterner::new)
    }
}

impl<T: Write + Read + Seek + ?Sized> IOInterner<T> {
    /// Like [`Self::get_or_intern`] instead that will return also a `bool` indicating whether the
    /// entry was not there before.
    /// 
    /// # Errors
    /// 
    /// See [`Self::get_or_intern`].
    pub fn try_intern<U: Read + Seek>(&self, mut buf: U) -> io::Result<(IOObj<'_, T>, bool)> {
        let mut l = self.inner.lock().unwrap();

        let buf_len = buf.seek(SeekFrom::End(0))?;

        if buf_len == 0 {
            return Ok((IOObj {
                start_init: 0,
                start: 0,
                len: 0,
                guard: &self.inner,
            }, false));
        }

        let len = l.seek(SeekFrom::End(0))?;

        let mut pos = IOPos {
            start_pos: len,
            len: buf_len,
        };

        let mut temp_buf = Vec::new();

        if len == buf_len {
            if eq(&mut *l, &mut buf)? {
                pos.start_pos = 0;
                return Ok((self.get_pos(pos), false))
            }
        } else if buf_len < len {
            let mut intern_buf = VecDeque::new();

            buf.seek(SeekFrom::Start(0))?;
            Read::take(&mut buf, buf_len).read_to_end(&mut temp_buf)?;

            intern_buf.reserve(buf_len as _);

            l.seek(SeekFrom::Start(0))?;
            io_unary_op(Read::take(&mut *l, buf_len), |x| { intern_buf.extend(x); false })?;

            let d = len - buf_len;

            let mut f = |start, intern_buf: &VecDeque<u8>| {
                let (front1, back1) = intern_buf.as_slices();
                let (front2, back2) = temp_buf.split_at(front1.len());

                if front1 == front2 && back1 == back2 {
                    pos.start_pos = start;
                    Some((self.get_pos(pos), false))
                } else {
                    None
                }
            };

            let mut read_buf = [0; BUF_LEN];
            let mut index = 0;

            io::copy(&mut Read::take(&mut *l, BUF_LEN as _), &mut &mut read_buf[..])?;

            for start in 0..d {
                if let Some(e) = f(start, &intern_buf) {
                    return Ok(e)
                }

                intern_buf.pop_front();
                intern_buf.push_back(read_buf[index]);

                index += 1;

                if index == BUF_LEN {
                    io::copy(&mut Read::take(&mut *l, BUF_LEN as _), &mut &mut read_buf[..])?;
                    index = 0;
                }
            }

            if let Some(e) = f(d, &intern_buf) {
                return Ok(e)
            }
        }

        if temp_buf.is_empty() {
            io::copy(&mut buf, &mut *l)?;
        } else {
            l.write_all(&temp_buf)?;
        }

        l.flush().map(|_| (self.get_pos(pos), true))
    }

    /*
    other implementation that does not allocate nothing but benchmarked horrible with files
    pub fn try_intern<U: Read + Seek>(&self, mut buf: U) -> io::Result<(IOObj<'_, T>, bool)> {
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
            buf.seek(SeekFrom::Start(0))?;
            l.seek(SeekFrom::Start(start))?;

            if starts_with(&mut *l, &mut buf)? {
                pos.start_pos = start;
                return Ok((self.get_pos(pos), false))
            }

        }

        l.seek(SeekFrom::End(0))?;
            buf.seek(SeekFrom::Start(0))?;

        io::copy(&mut buf, &mut *l)?;
        l.flush()?;
        Ok((self.get_pos(pos), true))
    }
    */

    /// Creates a new `IOEntry` object that will be able to generate [`IOObj`] that always read
    /// same bytes as `buf` from `T` so `buf` will be written at the end if it's not already.
    ///
    /// The [`IOObj`] does not implement `Write` as that would mess up equality of two
    /// [`IOEntry`] instances pointing to the same position with same length which it's actually
    /// the comparison algorithm.
    /// 
    /// The cursor of the inner [`Mutex<T>`] field it's setted at final of the [`IOPos`] returned
    /// which would be at final if the data had to be written.
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
    /// ## Race conditions
    /// 
    /// if a `&File` it's passed down as `buf`,the length of the file it's readed at init and while
    /// comparing the contents new bytes added thought another `&File` gotta be ignored.
    /// 
    /// [`seek`][`Seek::seek`]
    /// [`write`][`Write::write`]
    /// [`read`][`Read::read`]
    pub fn get_or_intern<U: Read + Seek>(&self, buf: U) -> io::Result<IOObj<'_, T>> {
        self.try_intern(buf).map(|e| e.0)
    }

    /// Convenience for `self.get_or_intern(io::Cursor::new(bytes))`.
    pub fn get_or_intern_bytes(&self, bytes: impl AsRef<[u8]>) -> io::Result<IOObj<'_, T>> {
        self.get_or_intern(Cursor::new(bytes))
    }

    /// Convenience for `self.try_intern(io::Cursor::new(bytes))`.
    pub fn try_intern_bytes(&self, bytes: impl AsRef<[u8]>) -> io::Result<(IOObj<'_, T>, bool)> {
        self.try_intern(Cursor::new(bytes))
    }

    /// Creates an [`IOObj`] out it's first [`IOPos`].
    #[inline]
    pub fn get_pos(&self, pos: IOPos) -> IOObj<'_, T> {
        IOObj {
            start_init: pos.start_pos,
            start: pos.start_pos,
            len: pos.len,
            guard: &self.inner,
        }
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
        drop(l);
        self.seek(SeekFrom::Current(len as _))?;
        Ok(len)
    }
}

impl<'a, T: Read + Seek + ?Sized> PartialEq for IOObj<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        if ptr::eq(self.guard, other.guard) {
            self.position() == other.position()
        } else {
            eq(self.clone(), other.clone()).expect("io error while testing for equality")
        }
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

/*
r"(?x)
^
([Mon|Tue|Wed|Thu|Fri|Sat|Sun]),
\s+
(\d{2})
\s+
(\d{2})
\s+
(\d{4})
\s+
\d{2}
:
\d{2}
:
\d{2}
\s
GMT
$"
*/

/// Struct stating which data [`IOObj`] will read from the [`IOInterner`] internal IO object.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy, Default)]
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
pub fn io_op<R1: Read, R2: Read>(
    mut x: R1,
    mut y: R2,
    mut callback: impl FnMut(&[u8], &[u8]) -> bool,
) -> io::Result<()> {
    let mut buf1 = [0; BUF_LEN];
    let mut buf2 = [0; BUF_LEN];

    Ok(loop {
        let mut buf1r = &mut buf1[..];
        let mut buf2r = &mut buf2[..];

        let mut x = Read::take(&mut x, buf1r.len() as _);
        let mut y = Read::take(&mut y, buf1r.len() as _);

        let readed1 = io::copy(&mut x, &mut buf1r)? as usize;
        let readed2 = io::copy(&mut y, &mut buf2r)? as usize;

        if callback(&buf1[..readed1], &buf2[..readed2]) {
            break;
        }

        if readed1 == 0 || readed2 == 0 {
            break;
        }
    })
}

/// Checks if the contents of the reader `x` are the ones of `y`.
///
/// # Errors
///
/// See [`io_op`].
pub fn eq<R1: Read, R2: Read>(x: R1, y: R2) -> io::Result<bool> {
    let mut result = true;
    io_op(x, y, |x, y| { result = x == y; !result }).map(|_| result)
}

/// Checks if the first contents of the reader `haystack` are the ones of `needle`,an empty needle
/// is always true.
///
/// # Errors
///
/// See [`io_op`].
pub fn starts_with<R1: Read, R2: Read>(haystack: R1, needle: R2) -> io::Result<bool> {
    let mut result = true;
    io_op(haystack, needle, |x, y| { result = x.starts_with(y); !result }).map(|_| result)
}

/// Checks if the contents of `needle` are in `haystack`.
/// 
/// # Errors
/// 
/// See [`io_op`].
pub fn find<R1: Read + Seek, R2: Read + Seek>(mut haystack: R1, needle: R2) -> io::Result<Option<IOPos>> {
        let l = &mut haystack;
        let mut buf = needle;

        let buf_len = buf.seek(SeekFrom::End(0))?;

        let mut pos = IOPos::default();

        if buf_len == 0 {
            return Ok(Some(pos));
        }

        pos.len = buf_len;

        let len = l.seek(SeekFrom::End(0))?;

        if len == buf_len {
            eq(&mut *l, &mut buf).map(|x| bool::then(x, || pos))
        } else if buf_len < len {
            let mut temp_buf = Vec::new();
            let mut intern_buf = VecDeque::new();

            buf.seek(SeekFrom::Start(0))?;
            Read::take(&mut buf, buf_len).read_to_end(&mut temp_buf)?;

            intern_buf.reserve(buf_len as _);

            l.seek(SeekFrom::Start(0))?;
            io_unary_op(Read::take(&mut *l, buf_len), |x| { intern_buf.extend(x); false })?;

            let d = len - buf_len;

            let mut f = |start, intern_buf: &VecDeque<u8>| {
                let (front1, back1) = intern_buf.as_slices();
                let (front2, back2) = temp_buf.split_at(front1.len());

                if front1 == front2 && back1 == back2 {
                    pos.start_pos = start;
                    Some(pos)
                } else {
                    None
                }
            };

            let mut read_buf = [0; BUF_LEN];
            let mut index = 0;

            io::copy(&mut Read::take(&mut *l, BUF_LEN as _), &mut &mut read_buf[..])?;

            for start in 0..d {
                let x = f(start, &intern_buf);
                if x.is_some() {
                    return Ok(x)
                }

                intern_buf.pop_front();
                intern_buf.push_back(read_buf[index]);

                index += 1;

                if index == BUF_LEN {
                    io::copy(&mut Read::take(&mut *l, BUF_LEN as _), &mut &mut read_buf[..])?;
                    index = 0;
                }
            }

            Ok(f(d, &intern_buf))
        } else {
            Ok(None)
        }
}


/// Checks if the contents of `needle` are in `haystack` starting from the end.
/// 
/// # Errors
/// 
/// See [`io_op`].
pub fn rfind<R1: Read + Seek, R2: Read + Seek>(mut haystack: R1, needle: R2) -> io::Result<Option<IOPos>> {
        let l = &mut haystack;
        let mut buf = needle;

        let buf_len = buf.seek(SeekFrom::End(0))?;

        let mut pos = IOPos::default();

        if buf_len == 0 {
            return Ok(Some(pos));
        }

        pos.len = buf_len;

        let len = l.seek(SeekFrom::End(0))?;

        if len == buf_len {
            eq(&mut *l, &mut buf).map(|x| bool::then(x, || pos))
        } else if buf_len < len {
            let mut temp_buf = Vec::new();
            let mut intern_buf = VecDeque::new();

            buf.seek(SeekFrom::Start(0))?;
            Read::take(&mut buf, buf_len).read_to_end(&mut temp_buf)?;

            intern_buf.reserve(buf_len as _);

            let d = len - buf_len;

            l.seek(SeekFrom::Start(d))?;
            io_unary_op(Read::take(&mut *l, buf_len), |x| { intern_buf.extend(x); false })?;

            let mut f = |start, intern_buf: &VecDeque<u8>| {
                let (front1, back1) = intern_buf.as_slices();
                let (front2, back2) = temp_buf.split_at(front1.len());

                if front1 == front2 && back1 == back2 {
                    pos.start_pos = start;
                    Some(pos)
                } else {
                    None
                }
            };

            let mut cursor = d.saturating_sub(BUF_LEN as _);
            let mut read_buf = [0; BUF_LEN];
            l.seek(SeekFrom::Start(cursor))?;
            let mut index = io::copy(&mut Read::take(&mut *l, BUF_LEN as _), &mut &mut read_buf[..])? as usize;

            for start in (0..d).rev() {
                let x = f(start, &intern_buf);
                if x.is_some() {
                    return Ok(x)
                }

                intern_buf.push_front(read_buf[index]);
                intern_buf.pop_back();

                index = if let Some(e) = index.checked_sub(1) {
                    e
                } else {
                    l.seek(SeekFrom::Start(cursor))?;
                    cursor = d.saturating_sub(BUF_LEN as _);
                    io::copy(&mut Read::take(&mut *l, BUF_LEN as _), &mut &mut read_buf[..])? as _
                };
            }

            Ok(f(d, &intern_buf))
        } else {
            Ok(None)
        }
}
/*
pub fn contains<R1: Read, R2: Read>(mut haystack: R1, mut needle: R2) -> io::Result<bool> {
        let mut len = 0;
        let mut buf_len = 0;
        let mut temp_buf = Vec::new();

        let mut result = true;

        io_op(&mut haystack, &mut needle, |x, y| {
            len += x.len() as u64;
            buf_len += y.len() as u64;
            result = x == y;

            !result
        })?; 

        if len == buf_len {
            if eq(haystack, buf)? {
                return Ok(true)
            }
        } else if buf_len < len {
        let mut intern_buf = VecDeque::new();
            Read::take(&mut needle, buf_len).read_to_end(&mut temp_buf)?;

            intern_buf.reserve(buf_len as _);

            io_unary_op(Read::take(&mut haystack, buf_len), |x| { intern_buf.extend(x); false })?;

            let d = len - buf_len;

            let mut f = |intern_buf: &VecDeque<u8>| {
                let (front1, back1) = intern_buf.as_slices();
                let (front2, back2) = temp_buf.split_at(front1.len());

                front1 == front2 && back1 == back2
            };

            let mut read_buf = [0; BUF_LEN];
            let mut index = 0;

            io::copy(&mut Read::take(&mut haystack, BUF_LEN as _), &mut &mut read_buf[..])?;

            for _ in 0..d {
                intern_buf.pop_front();
                intern_buf.push_back(read_buf[index]);

                if f(&intern_buf) {
                    return Ok(true)
                }

                index += 1;

                if index == BUF_LEN {
                    io::copy(&mut Read::take(&mut haystack, BUF_LEN as _), &mut &mut read_buf[..])?;
                    index = 0;
                }
            }
        }

        Ok(false)
}
*/

/// Compares the contents of `x` to the ones of `y`,see
/// [`lexicographical comparison`][`Ord#lexicographical-comparison`].
///
/// # Errors
///
/// See [`io_op`].
pub fn cmp<R1: Read, R2: Read>(x: R1, y: R2) -> io::Result<Ordering> {
    let mut result = None;

    io_op(x, y, |x, y| {
        result = match x.cmp(y) {
            Equal => None,
            x => Some(x),
        };

        result.is_none()
    })
    .map(|_| result.unwrap_or(Equal))
}

/// Hash the contents of the reader `x` to `state`.
///
/// # Errors
///
/// See [`io_unary_op`].
pub fn hash<R1: Read, H: Hasher>(x: R1, state: &mut H) -> io::Result<()> {
    let mut len = 0;

    io_unary_op(x, |x| {
        Hash::hash_slice(x, state);
        len += x.len();
        false
    })?;

    len.hash(state);

    Ok(())
}

/// Convert the contents of the `reader` into valid UTF-8 and pass it to callback,passing in `None`
/// when some bytes are not.
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

            false
        })
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
pub fn io_unary_op<R1: Read>(
    mut x: R1,
    mut callback: impl FnMut(&[u8]) -> bool,
) -> io::Result<()> {

    let mut buf1 = [0; BUF_LEN];

    Ok(loop {
        let mut buf1r = &mut buf1[..];

        let mut x = Read::take(&mut x, buf1r.len() as _);

        let readed1 = io::copy(&mut x, &mut buf1r)? as usize;

        if readed1 == 0 || callback(&buf1[..readed1]) {
            break;
        }
    })
}

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

#[cfg(feature = "serde_support")]
pub mod serde {
    use super::*;

    impl<T: Write + Read + Seek + Serialize> Serialize for IOInterner<T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.inner.serialize(serializer)
        }
    }

    impl<'a, T: Write + Read + Seek + Deserialize<'a>> Deserialize<'a> for IOInterner<T> {
        fn deserialize<D: Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
            <Mutex<T>>::deserialize(deserializer).map(Self::from_inner)
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

            let mut result = None;

            io_unary_op(x, |x| {
                result = serializer.serialize_element(x).err();

                result.is_some()
            })?;

            break result.map(Err).unwrap_or_else(|| serializer.end())
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

    impl<'a, T: Seek + Read> Serialize for IOObj<'a, T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serialize(self.clone(), serializer).expect("io error while serializing")
        }
    }
}