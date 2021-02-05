
//! A simple crate implementing a struct wrapping a reader and writer that's used to create readers
//! to unique data.

use ::{
    std::{
        hash::{Hash, Hasher},
        io::{self, SeekFrom, prelude::*},
        ptr,
        sync::Mutex,
    },
};

/// A struct wrapping a `Mutex<T>` used for storing and retrieving data thought readers.
/// 
/// Note that `T` it's wrapped into a [`Mutex`] for ensure [`IOObj`] does not lock access to the
/// `IOInterner` and to guarantee will only lock at `Read` methods.
pub struct IOInterner<T: Write + Read + Seek>(Mutex<T>);

/// A struct generated by [`IOInterner::get_or_intern`].
pub struct IOObj<'a, T> {
    start_init: u64,
    start: u64,
    len: u64,
    guard: &'a Mutex<T>
}

impl<'a, T> PartialEq for IOObj<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.start_init == other.start_init && self.start == other.start && ptr::eq(self.guard, other.guard)
    }
}

impl<'a, T> Clone for IOObj<'a, T> {
    fn clone(&self) -> Self {
        Self {
            start_init: self.start_init,
            start: self.start,
            len: self.len,
            guard: self.guard
        }
    }
}

impl<'a, T> Eq for IOObj<'a, T> {}

impl<'a, T> Hash for IOObj<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start_init.hash(state);
        self.start.hash(state);
        ptr::hash(self.guard, state)
    }
}

impl<'a, T: Read + Seek> Read for IOObj<'a, T> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.len == 0 {
            return Ok(0)
        }

        let mut l = self.guard.lock().unwrap();
        l.seek(SeekFrom::Start(self.start))?;

        let len = <&mut T as Read>::take(&mut *l, self.len).read(buf)?;
        self.seek(SeekFrom::Current(len as _))?;
        Ok(len)
    }
}

impl<'a, T> Seek for IOObj<'a, T> {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {

        match pos {
            SeekFrom::Current(x) => {
                if x.is_negative() { 
                    let x = x.abs() as u64;
                    let new_start = self.start.saturating_sub(x);

                    if new_start < self.start_init {
                        return Err(io::ErrorKind::InvalidInput.into())
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
                let initial_len = self.len + (self.start - self.start_init);

                self.seek(SeekFrom::Start(if x.is_negative() { 
                    let x = x.abs() as u64;
                    let new_start = initial_len.checked_sub(x);

                    if new_start.is_none() {
                        return Err(io::ErrorKind::InvalidInput.into())
                    }

                    new_start.unwrap()
                } else {
                    initial_len
                }))
            }
        }
    }
}

impl<T: Write + Read + Seek> IOInterner<T> {
    /// Create a new `IOInterner` that uses as common storage `x`.
    pub fn new(x: T) -> Self {
        Self(Mutex::new(x))
    }

    /// Creates a new [`IOObj`] object that will always read same bytes as `buf` from `T` so `buf`
    /// will be written if it's not already.
    /// 
    /// The `IOObj` returned does not implement `Write` as that would mess up uniqueness that allow
    /// comparisons between it only have to look at the starting position,the advanced cursor and
    /// the pointer to guarantee the interner it's the same.
    pub fn get_or_intern<U: Read + Seek>(&self, mut buf: U) -> io::Result<IOObj<'_, T>> {
        let mut l = self.0.lock().unwrap();

        let buf_len = buf.seek(SeekFrom::End(0))?;

        if buf_len == 0 {
            return Ok(
                IOObj {
                    start_init: 0,
                    start: 0,
                    len: 0,
                    guard: &self.0              
                }
            )
        }

        let len = l.seek(SeekFrom::End(0))?;

        for start in 0..len.saturating_sub(buf_len) {
            l.seek(SeekFrom::Start(start))?;
            buf.seek(SeekFrom::Start(0))?;

            if starts_with(&mut *l, &mut buf)? {
                return Ok(IOObj {
                        start_init: start,
                        start,
                        len: buf_len,
                        guard: &self.0,
                    })
            }
        }

        l.seek(SeekFrom::Start(len))?;
        buf.seek(SeekFrom::Start(0))?;
        io::copy(&mut buf, &mut *l)?;

        Ok(IOObj {
            start_init: len,
            start: len,
            len: buf_len,
            guard: &self.0,
        })
    }
}

/// Checks if the contents of the reader `x` are the ones of `y`.
/// 
/// # Errors
/// 
/// See [`io::copy`].
pub fn eq<R1: Read, R2: Read>(x: R1, y: R2) -> io::Result<bool> {
    io_op(x, y, PartialEq::eq)
}

/// Checks if the first contents of the reader `haystack` are the ones of `needle`,an empty needle
/// is always true.
/// 
/// # Errors
/// 
/// See [`io::copy`].
pub fn starts_with<R1: Read, R2: Read>(haystack: R1, needle: R2) -> io::Result<bool> {
    io_op(haystack, needle, <[u8]>::starts_with)
}

fn io_op<R1: Read, R2: Read>(mut x: R1, mut y: R2, callback: impl Fn(&[u8], &[u8]) -> bool) -> io::Result<bool> {
    let mut buf1 = [0; 512];
    let mut buf2 = [0; 512];

    Ok(loop {
        let mut buf1 = &mut buf1[..];
        let mut buf2 = &mut buf2[..];

        let mut x = (&mut x).take(buf1.len() as _);
        let mut y = (&mut y).take(buf1.len() as _);

        let readed1 = io::copy(&mut x, &mut buf1)? as usize;
        let readed2 = io::copy(&mut y, &mut buf2)? as usize;

        if !callback(&buf1[..readed1], &buf2[..readed2]) {
            break false
        }

        if readed1 == 0 || readed2 == 0 {
            break true
        }
    })
}