"""
data.py — Universe, price, and benchmark loaders for the Nifty Momentum app.

Caching strategy on Streamlit Cloud:
    - A seed parquet (cache/prices_seed.parquet) ships with the repo.
    - On first request, we load the seed into memory (st.cache_data).
    - If the user's end_date extends past the seed's last date, we fetch
      the delta from yfinance for symbols that need it.
    - All updates stay in the in-memory cache; we never rely on disk writes
      persisting across container restarts.

This keeps the user experience snappy on Cloud (instant first load from seed)
while still allowing the app to pull fresh data when requested.
"""

from __future__ import annotations

from pathlib import Path
import time
import warnings
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

UNIVERSE_DIR = Path("universes")
CACHE_DIR = Path("cache")
PRICES_SEED = CACHE_DIR / "prices_seed.parquet"
BENCHMARKS_SEED = CACHE_DIR / "benchmarks_seed.parquet"

# Universe -> (CSV filename, yfinance benchmark ticker, fallback ticker)
UNIVERSE_CONFIG = {
    "Nifty 50":  ("ind_nifty50list.csv",  "^NSEI",      "^NSEI"),
    "Nifty 200": ("ind_nifty200list.csv", "^CRSLDX200", "^NSEI"),
    "Nifty 500": ("ind_nifty500list.csv", "^CRSLDX",    "^NSEI"),
}


# ---------------------------------------------------------------------------
# UNIVERSE
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_universe(universe_name: str) -> pd.DataFrame:
    """Read the constituent CSV for a named index. Returns symbol + sector."""
    cfg = UNIVERSE_CONFIG.get(universe_name)
    if cfg is None:
        raise ValueError(f"Unknown universe: {universe_name}")
    csv_path = UNIVERSE_DIR / cfg[0]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} missing. The app needs constituent CSVs in {UNIVERSE_DIR}/"
        )
    df = pd.read_csv(csv_path)
    df["yf_symbol"] = df["Symbol"].astype(str).str.strip() + ".NS"
    df = df.rename(columns={"Industry": "sector"})
    return df[["yf_symbol", "sector"]]


def get_benchmark_ticker(universe_name: str) -> tuple[str, str]:
    """Return (primary, fallback) benchmark tickers for the universe."""
    cfg = UNIVERSE_CONFIG[universe_name]
    return cfg[1], cfg[2]


# ---------------------------------------------------------------------------
# PRICES — seed-first strategy
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_price_seed() -> pd.DataFrame:
    """Load the bundled price seed parquet. Returns empty DataFrame if absent."""
    if PRICES_SEED.exists():
        df = pd.read_parquet(PRICES_SEED)
        df["date"] = pd.to_datetime(df["date"])
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["close", "open"])
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns the date as index, sometimes named oddly. Normalize."""
    df = df.reset_index()
    df.columns = [c.lower() if not isinstance(c, tuple) else c[0].lower() for c in df.columns]
    if "date" not in df.columns:
        for cand in ("index", "datetime", "timestamp"):
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
    return df


def fetch_delta_prices(
    symbols: list[str],
    start: str,
    end: str,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Download OHLCV for the given symbols + date range from yfinance.
    Used when seed doesn't cover the requested window.

    progress_callback(done, total, message) is called per chunk.

    Resilience:
      - chunked at 25 symbols per call (smaller = less throttle pressure)
      - retries each chunk up to 3 times with exponential backoff
      - tracks failed symbols and re-attempts at the end individually
    """
    import yfinance as yf

    out_frames: list[pd.DataFrame] = []
    failed_symbols: list[str] = []
    chunk_size = 25
    n_chunks = (len(symbols) - 1) // chunk_size + 1
    max_retries = 3

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        chunk_idx = i // chunk_size + 1
        if progress_callback:
            progress_callback(chunk_idx, n_chunks,
                              f"downloading chunk {chunk_idx}/{n_chunks}")

        chunk_succeeded = False
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    chunk, start=start, end=end,
                    group_by="ticker", auto_adjust=True,
                    progress=False, threads=True,
                )
                if df is None or df.empty:
                    raise ValueError("empty response")

                # Extract per-symbol frames
                got_count = 0
                for sym in chunk:
                    if sym not in df.columns.get_level_values(0):
                        continue
                    sub = df[sym].copy()
                    sub = sub.reset_index()
                    sub.columns = [str(c).lower() for c in sub.columns]
                    if "date" not in sub.columns:
                        for cand in ("index", "datetime", "timestamp"):
                            if cand in sub.columns:
                                sub = sub.rename(columns={cand: "date"})
                                break
                    if "close" not in sub.columns:
                        continue
                    sub = sub.dropna(subset=["close"])
                    if sub.empty:
                        continue
                    sub["symbol"] = sym
                    out_frames.append(sub)
                    got_count += 1
                chunk_succeeded = True
                if got_count < len(chunk) * 0.5:
                    print(f"[data] chunk {chunk_idx}: only {got_count}/{len(chunk)} symbols got data")
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"[data] chunk {chunk_idx} attempt {attempt + 1} failed: {e}. "
                      f"Waiting {wait}s...")
                time.sleep(wait)

        if not chunk_succeeded:
            failed_symbols.extend(chunk)
            print(f"[data] chunk {chunk_idx} permanently failed")

        time.sleep(0.8)  # be polite to yfinance between chunks

    # Final retry pass: try failed symbols one at a time
    if failed_symbols:
        print(f"[data] retrying {len(failed_symbols)} failed symbols individually...")
        for j, sym in enumerate(failed_symbols):
            if progress_callback:
                progress_callback(
                    n_chunks, n_chunks,
                    f"retry {j + 1}/{len(failed_symbols)}: {sym}"
                )
            try:
                df = yf.download(sym, start=start, end=end,
                                 auto_adjust=True, progress=False)
                if df is None or df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.reset_index()
                df.columns = [str(c).lower() for c in df.columns]
                if "date" not in df.columns:
                    for cand in ("index", "datetime", "timestamp"):
                        if cand in df.columns:
                            df = df.rename(columns={cand: "date"})
                            break
                if "close" not in df.columns:
                    continue
                df = df.dropna(subset=["close"])
                if df.empty:
                    continue
                df["symbol"] = sym
                out_frames.append(df)
            except Exception as e:
                print(f"[data] {sym} final retry failed: {e}")
            time.sleep(0.3)

    if not out_frames:
        return pd.DataFrame()

    out = pd.concat(out_frames, ignore_index=True)
    out = out.dropna(subset=["close"])
    print(f"[data] total: {out['symbol'].nunique()} symbols, {len(out):,} rows")
    return out


def get_prices(
    symbols: list[str],
    start: str,
    end: str,
    progress_callback=None,
    use_seed_only: bool = False,
) -> pd.DataFrame:
    """
    Resolve prices for symbols and date range.

    Strategy (incremental, per-symbol):
      1. Load the seed.
      2. For each requested symbol, find its latest cached date.
      3. Symbols whose latest date >= end_date are FULLY satisfied — skip.
      4. Symbols whose latest date < end_date need a gap-fill (fetch only the
         missing tail for THAT symbol).
      5. After fetch, persist the merged data back to the seed parquet so the
         next run benefits from the new data.

    With use_seed_only=True, step 4 is skipped entirely.
    """
    seed = load_price_seed()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    have = seed[seed["symbol"].isin(symbols)] if not seed.empty else seed.iloc[:0]

    # Per-symbol latest cached date
    if not have.empty:
        latest_per_symbol = have.groupby("symbol")["date"].max().to_dict()
    else:
        latest_per_symbol = {}

    # Partition symbols: which need updating, which are current
    # Key insight: skip the fetch if the gap contains no NSE trading days.
    # We approximate trading days as business days (Mon-Fri). This will
    # occasionally trigger an unnecessary fetch on Indian market holidays
    # that fall on weekdays — minor cost, much simpler than full holiday calendar.
    symbols_to_fetch: dict[str, str] = {}    # symbol -> per-symbol fetch_start (ISO)
    for sym in symbols:
        last = latest_per_symbol.get(sym)
        if last is None:
            # Never cached -> need full range from start
            if not use_seed_only:
                symbols_to_fetch[sym] = start
            continue

        if last >= end_ts:
            continue  # already covers requested window

        # Count business days strictly between last_date and end_date.
        # If there are no business days in that gap, the symbol is effectively
        # up-to-date — NSE was closed.
        gap_start = (last + pd.Timedelta(days=1)).normalize()
        gap_end = end_ts.normalize()
        if gap_start > gap_end:
            continue
        # bdate_range is inclusive on both ends; counts Mon-Fri
        n_business_days = len(pd.bdate_range(start=gap_start, end=gap_end))
        if n_business_days == 0:
            continue  # nothing to fetch — gap is just weekend(s)

        if not use_seed_only:
            symbols_to_fetch[sym] = gap_start.strftime("%Y-%m-%d")

    if symbols_to_fetch:
        n_to_fetch = len(symbols_to_fetch)
        n_current = len(symbols) - n_to_fetch
        if progress_callback:
            progress_callback(0, 1,
                              f"{n_current} symbols up to date, "
                              f"fetching gaps for {n_to_fetch}...")
        print(f"[data] {n_current}/{len(symbols)} symbols current; "
              f"fetching gaps for {n_to_fetch}")

        # If there are multiple distinct fetch starts among stale symbols,
        # consolidate by using the EARLIEST start across all of them.
        # We download a slightly wider window for some symbols (cheap) so
        # everything goes through ONE batched fetch instead of N separate ones.
        # Drop-duplicates at merge time handles any overlap with seed.
        earliest_start = min(symbols_to_fetch.values())
        syms_to_fetch = list(symbols_to_fetch.keys())
        print(f"[data] consolidated fetch: {len(syms_to_fetch)} symbols from {earliest_start} to {end}")

        delta_frames = []
        if pd.Timestamp(earliest_start) <= end_ts:
            d = fetch_delta_prices(syms_to_fetch, earliest_start, end, progress_callback)
            if not d.empty:
                delta_frames.append(d)

        if delta_frames:
            delta = pd.concat(delta_frames, ignore_index=True)
            # Merge into seed and persist
            full_seed = seed if not seed.empty else pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "symbol"]
            )
            combined_seed = pd.concat([full_seed, delta], ignore_index=True)
            combined_seed = combined_seed.drop_duplicates(
                subset=["date", "symbol"], keep="last"
            )
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                combined_seed.to_parquet(PRICES_SEED)
                print(f"[data] updated seed: {len(combined_seed):,} rows, "
                      f"{combined_seed['symbol'].nunique()} symbols")
                # Bust the @st.cache_data so subsequent calls see the updated seed
                load_price_seed.clear()
            except Exception as e:
                print(f"[data] could not persist seed update: {e}")

            # Use the combined data going forward in this run
            have = combined_seed[combined_seed["symbol"].isin(symbols)]

    # Return the slice that matches the requested window
    if have.empty:
        return have
    return have[(have["date"] >= start_ts) & (have["date"] <= end_ts)].copy()


# ---------------------------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_benchmark_seed() -> pd.DataFrame:
    """Load bundled benchmark seed."""
    if BENCHMARKS_SEED.exists():
        df = pd.read_parquet(BENCHMARKS_SEED)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame(columns=["date", "ticker", "close"])


def get_benchmark(
    universe_name: str,
    start: str,
    end: str,
    use_seed_only: bool = False,
) -> tuple[pd.Series, str]:
    """
    Return (close_series, ticker_used) for the universe's benchmark.
    Falls back gracefully if primary ticker has no data.

    When use_seed_only=True, never call yfinance — return whatever the seed has.
    """
    import yfinance as yf
    primary, fallback = get_benchmark_ticker(universe_name)
    seed = load_benchmark_seed()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    tried = []
    for ticker in (primary, fallback):
        if ticker in tried:
            continue
        tried.append(ticker)

        # Check seed first
        sub = seed[seed["ticker"] == ticker] if not seed.empty else pd.DataFrame()

        if not sub.empty:
            seed_max = sub["date"].max()
            # Seed fully covers — return slice directly
            if seed_max >= end_ts:
                s = sub[(sub["date"] >= start_ts) & (sub["date"] <= end_ts)]
                if not s.empty:
                    print(f"[bench] {ticker}: served from seed ({len(s)} rows)")
                    return s.set_index("date")["close"].sort_index(), ticker

            # Seed has partial data + seed-only mode: return what we have
            if use_seed_only:
                s = sub[(sub["date"] >= start_ts) & (sub["date"] <= end_ts)]
                if not s.empty:
                    print(f"[bench] {ticker}: partial from seed ({len(s)} rows, "
                          f"clamped to {seed_max.date()})")
                    return s.set_index("date")["close"].sort_index(), ticker
                continue

            # Seed has partial data — fetch only the gap
            gap_start = (seed_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"[bench] {ticker}: seed has data through {seed_max.date()}, "
                  f"fetching {gap_start} to {end}")
            fetch_start_str = gap_start
        else:
            # No seed data for this ticker
            if use_seed_only:
                continue
            fetch_start_str = start

        # Need to fetch from yfinance
        try:
            df = yf.download(ticker, start=fetch_start_str, end=end,
                             auto_adjust=True, progress=False)
        except Exception as e:
            print(f"[bench] {ticker} download exception: {e}")
            continue

        if df is None or df.empty:
            # If yfinance failed but we have partial seed data, use that
            if not sub.empty:
                s = sub[(sub["date"] >= start_ts) & (sub["date"] <= end_ts)]
                if not s.empty:
                    print(f"[bench] {ticker}: yfinance empty, using partial seed")
                    return s.set_index("date")["close"].sort_index(), ticker
            print(f"[bench] {ticker}: yfinance returned empty")
            continue

        # Flatten potential MultiIndex columns: ('Close', '^NSEI') -> 'close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]

        # Normalize the date column name
        if "date" not in df.columns:
            for cand in ("index", "datetime", "timestamp"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "date"})
                    break

        if "date" not in df.columns or "close" not in df.columns or df.empty:
            print(f"[bench] {ticker}: missing date/close cols after normalize. "
                  f"Got: {list(df.columns)}")
            continue

        s = df.set_index("date")["close"].sort_index().dropna()
        if s.empty:
            print(f"[bench] {ticker}: empty after dropna")
            continue

        print(f"[bench] {ticker}: fetched {len(s)} new rows from {fetch_start_str}")

        # Combine fetched delta with existing seed data for this ticker
        if not sub.empty:
            existing_s = sub.set_index("date")["close"].sort_index()
            combined_s = pd.concat([existing_s, s])
            combined_s = combined_s[~combined_s.index.duplicated(keep="last")].sort_index()
        else:
            combined_s = s

        combined_s.name = ticker

        # Persist to seed for next run
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            new_rows = pd.DataFrame({
                "date": combined_s.index,
                "ticker": ticker,
                "close": combined_s.values,
            })
            existing = load_benchmark_seed()
            # Drop old rows for this ticker, then append updated
            if not existing.empty:
                other = existing[existing["ticker"] != ticker]
                merged = pd.concat([other, new_rows], ignore_index=True)
            else:
                merged = new_rows
            merged.to_parquet(BENCHMARKS_SEED)
            print(f"[bench] persisted {len(merged)} total rows to seed")
            load_benchmark_seed.clear()
        except Exception as e:
            print(f"[bench] persist failed: {e}")

        # Return only the requested window
        windowed = combined_s[(combined_s.index >= start_ts) & (combined_s.index <= end_ts)]
        return windowed, ticker

    raise RuntimeError(
        f"No benchmark data fetched for {universe_name}. "
        f"Tried tickers: {tried}. "
        f"{'Seed-only mode: ensure your seed contains one of these tickers.' if use_seed_only else 'yfinance is rate-limiting or the index ticker is broken. Try seed-only mode or wait 5 minutes.'}"
    )
