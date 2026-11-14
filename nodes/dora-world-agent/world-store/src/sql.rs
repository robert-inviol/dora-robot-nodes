//! The agents' SQL: one statement on a read-only connection whose authorizer allows only reads, interrupted when it
//! runs past its time limit, cut at a row limit.

use std::num::NonZeroUsize;
use std::path::Path;
use std::time::{Duration, Instant};

use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, ErrorCode, OpenFlags};
use world_model::{SqlRows, SqlValue, WorldError};

/// How many virtual machine steps run between time checks.
const STEPS_PER_CHECK: i32 = 1000;

fn allow_reads_only(context: AuthContext<'_>) -> Authorization {
    match context.action {
        AuthAction::Select | AuthAction::Read { .. } | AuthAction::Function { .. } | AuthAction::Recursive => Authorization::Allow,
        _ => Authorization::Deny,
    }
}

/// Opens the database read-only, with the read-only authorizer installed.
pub fn open_guarded(path: &Path) -> rusqlite::Result<Connection> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
    connection.pragma_update(None, "query_only", true)?;
    connection.authorizer(Some(allow_reads_only));
    Ok(connection)
}

fn refused(error: rusqlite::Error, time_limit: Duration) -> WorldError {
    match &error {
        rusqlite::Error::SqliteFailure(failure, _) if failure.code == ErrorCode::OperationInterrupted => {
            WorldError::QueryTooSlow { limit_ms: time_limit.as_millis() as u64 }
        }
        rusqlite::Error::MultipleStatement => WorldError::BadQuery("only one statement is allowed".to_owned()),
        _ => WorldError::BadQuery(error.to_string()),
    }
}

fn value(cell: ValueRef<'_>) -> SqlValue {
    match cell {
        ValueRef::Null => SqlValue::Null,
        ValueRef::Integer(i) => SqlValue::Integer(i),
        ValueRef::Real(r) => SqlValue::Real(r),
        ValueRef::Text(t) => SqlValue::Text(String::from_utf8_lossy(t).into_owned()),
        ValueRef::Blob(b) => SqlValue::Text(format!("<{} byte blob>", b.len())),
    }
}

/// Runs one read-only statement on a guarded connection and returns at most `limit` rows.
pub fn run_read_only(connection: &Connection, sql: &str, limit: NonZeroUsize, time_limit: Duration) -> Result<SqlRows, WorldError> {
    let deadline = Instant::now() + time_limit;
    connection.progress_handler(STEPS_PER_CHECK, Some(move || Instant::now() >= deadline));
    let outcome = (|| {
        let mut statement = connection.prepare(sql)?;
        if !statement.readonly() {
            return Err(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_AUTH),
                Some("the statement writes".to_owned()),
            ));
        }
        let columns: Vec<String> = statement.column_names().into_iter().map(str::to_owned).collect();
        let width = columns.len();
        let mut rows = Vec::new();
        let mut truncated = false;
        let mut cursor = statement.query([])?;
        while let Some(row) = cursor.next()? {
            if rows.len() == limit.get() {
                truncated = true;
                break;
            }
            rows.push((0..width).map(|i| row.get_ref(i).map(value)).collect::<rusqlite::Result<Vec<_>>>()?);
        }
        Ok(SqlRows { columns, rows, truncated })
    })();
    connection.progress_handler(STEPS_PER_CHECK, None::<fn() -> bool>);
    outcome.map_err(|error| refused(error, time_limit))
}
