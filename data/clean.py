import polars as pl
import os
from datetime import datetime, timedelta
from functools import wraps
import time


# Decorators
def func_timer(func):
    @wraps(func)  # Best practice: preserves original function metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Call original function
        end_time = time.time()
        print(f"Calling function '{func.__name__}' time taken: {end_time - start_time:.2f} seconds.")
        return result
    return wrapper


def check_size(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(df_in := args[0], pl.DataFrame):
            print(f"DataFrame shape before '{func.__name__}': {df_in.shape}")
        df_out = func(*args, **kwargs)
        if isinstance(df_out, pl.DataFrame):
            print(f"DataFrame shape after '{func.__name__}': {df_out.shape}")
        return df_out
    return wrapper


# Cleaning Functions
@func_timer
def timezone_conversion(df: pl.DataFrame) -> pl.DataFrame:
    """
    Converts local time to UTC using the timezoneid column.
    Polars handles this efficiently with expressions.
    """
    # Polars allows per-row timezone conversion using 'replace_time_zone'
    # but for varied timezones, we map or use a loop over unique zones for speed.
    unique_tzs = df.select("timezoneid").unique().to_series()

    # Initialize a list to collect dataframes per timezone
    frames = []
    for tz in unique_tzs:
        # Filter for this timezone, localize, and convert to UTC
        temp = (
            df.filter(pl.col("timezoneid") == tz)
            .with_columns(
                pl.col("datetime_local")
                .str.to_datetime("%Y-%m-%d %H:%M:%S.000", strict=False)
                .dt.replace_time_zone(tz, ambiguous="null")
                .dt.convert_time_zone("UTC")
                .alias("datetime_utc")
            )
        )
        frames.append(temp)

    res = pl.concat(frames)

    return res.with_columns(
        pl.col("datetime_utc").dt.strftime("%Y-%m-%d").alias("date_utc")
    )


@func_timer
def join_data(path: str) -> pl.DataFrame:
    # Load past and current form, then union
    form_past = pl.read_csv(f"{path}/form_past.csv", infer_schema_length=10000)
    form_current = pl.read_csv(f"{path}/form_current.csv", infer_schema_length=10000)
    
    # Align schemas by casting all columns to the schema of form_past
    form_current = form_current.cast(form_past.schema)
    
    # Process datetime_utc for both form dataframes
    def process_form(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("datetime_utc").str.to_datetime("%Y-%m-%d %H:%M:%S.000", strict=False)
            .dt.replace_time_zone("UTC")
        )
    
    form_past = process_form(form_past)
    form_current = process_form(form_current)
    
    # Union the two form dataframes
    form = pl.concat([form_past, form_current], how="vertical")

    # Load past and current results, then union
    results_past = pl.read_csv(f"{path}/results_past.csv", infer_schema_length=100000)
    results_current = pl.read_csv(f"{path}/results_current.csv", infer_schema_length=100000)
    
    # Align schemas by casting all columns to the schema of results_past
    results_current = results_current.cast(results_past.schema)
    
    # Process timezone conversion for both
    results_past = timezone_conversion(results_past)
    results_current = timezone_conversion(results_current)
    
    # Union the two results dataframes
    results = pl.concat([results_past, results_current], how="vertical")

    # Prepare for join
    double_cols = ["date_utc", "racename", "distance", "trainer", "racegrade"]
    results_to_join = results.drop(double_cols)

    # Left Join
    train = form.join(
        results_to_join,
        on=["datetime_utc", "track", "dogname", "racenum"],
        how="left"
    )

    return train.with_columns(
        pl.col("datetime_utc").dt.strftime("%Y-%m-%d").alias("date")
    )

@func_timer
def missingness_check(df: pl.DataFrame) -> None:
    # Missingness checks for key columns
    dt_str = (datetime.utcnow() + timedelta(hours=10)).strftime("%Y-%m-%d")
    past_races = df.filter(pl.col("date") < dt_str)
    cols_to_check = ["box", "win_bsp", "place", "runtime", "speedc", "splitmargin", "margin1", "margin2"]
    for column in cols_to_check:
        if column not in past_races.columns:
            print(f"Column '{column}' not found in DataFrame.")
            continue
        missing_rate = past_races[column].is_null().mean()
        print(f"Missing {column}: {missing_rate:.2%}")


@check_size
def drop_no_box(df: pl.DataFrame) -> pl.DataFrame:
    dt_str = (datetime.utcnow() + timedelta(hours=10)).strftime("%Y-%m-%d")
    df = df.filter((pl.col("date") >= dt_str) | (pl.col("box").is_not_null()))
    df = df.with_columns(pl.col("box").cast(pl.Int32))
    return df


@check_size
def drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    df = df.unique()
    return df


@func_timer
def create_index(df: pl.DataFrame) -> pl.DataFrame:
    # Create Index
    df = df.with_columns(
        (pl.col("datetime_utc").cast(pl.String) + pl.col("track") +
         pl.col("racenum") + pl.col("quali").cast(pl.String)).alias("idx")
    )
    return df


@func_timer
def clean_track(df: pl.DataFrame) -> pl.DataFrame:
    # Track Cleaning & Basic Formatting
    df = df.with_columns([
        pl.col("track").replace({
            "straight": "",
            "dport@hob": "hobart",
            "dport@lcn": "launceston",
            "warrnamboolextra": "warrnambool"
        }),
        pl.col("racenum").cast(pl.String),
        pl.col("quali").fill_null(""),
    ])
    # Quali Flag
    df = df.with_columns([
        pl.col("quali").replace({"Q": "1", "": "0"}).fill_null("0").cast(pl.Int16)
    ])
    return df


@func_timer
def clean_race_features(df: pl.DataFrame) -> pl.DataFrame:
    # Numeric Cleaning
    df = df.with_columns([
        pl.col("distance").str.replace("m", "").replace("", "400").cast(pl.Float64),
        pl.col("handicap").replace({"Y": "1", "": "0"}).fill_null("0").cast(pl.Float64)
    ])
    # Fix Prizemoney (Regex replace $)
    for i in range(1, 9):
        col = f"prizemoney{i}"
        df = df.with_columns(
            pl.col(col).str.replace(r"\$", "").replace("", "0").cast(pl.Float64)
        )
    return df

@func_timer
def create_speedc(df: pl.DataFrame) -> pl.DataFrame:
        #### Speed ####
    # Calculate Speed. The column "speed" is from the form data prior to the race.
    # I don't know if it can be trusted, hence calculating our own speedc.
    df = df.with_columns((pl.col("distance") / pl.col("runtime")).alias("speedc"))

    # Median speed for correction (use 3rd place times)
    median_speeds = (
        df.filter(pl.col("place") == "3")
        .group_by("distance")
        .agg(pl.col("speedc").median().alias("med_speed"))
    )
    df = df.join(median_speeds, on="distance", how="left")

    # Runtime corrections (simplified syntax)
    df = df.with_columns(
        pl.when((pl.col("runtime") == 0) | pl.col("runtime").is_null())
        .then(pl.col("distance") / pl.col("med_speed"))
        .otherwise(pl.col("runtime"))
        .alias("runtime")
    )

    # Recalculate speedc with corrected runtime
    df = df.with_columns((pl.col("distance") / pl.col("runtime")).alias("speedc"))

    # Median splitmargin for correction (use 3rd place times)
    median_splitmargins = (
        df.filter(pl.col("place") == "3")
        .group_by("distance")
        .agg(pl.col("splitmargin").median().alias("med_splitmargin"))
    )

    median_splitmargins = median_splitmargins.fill_null(0)

    df = df.join(median_splitmargins, on="distance", how="left")

    df = df.with_columns(
        pl.when((pl.col("splitmargin") == 0) | pl.col("splitmargin").is_null())
        .then(pl.col("med_splitmargin"))
        .otherwise(pl.col("splitmargin"))
        .alias("splitmargin")
    )
    return df


def clean_margins(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("margin1").fill_null(1.0).alias("margin1"),
        pl.col("margin2").fill_null(1.0).alias("margin2"),
    )
    return df


def create_dog_form_features(df: pl.DataFrame) -> pl.DataFrame:
    # Stats from startsttd/startstot
    for c in ["startsttd", "startstot"]:
        prefix = c.replace("starts", "")
        # Extracting components using regex capture
        df = df.with_columns([
            pl.col(c).str.extract(r" (\d+)-", 1).fill_null("0").cast(pl.Int32).alias(f"runs_{prefix}"),
            pl.col(c).str.extract(r"-(\d+)-", 1).fill_null("0").cast(pl.Int32).alias(f"wins_{prefix}"),
            pl.col(c).str.extract(r"-(\d+)-(\d+)-", 1).fill_null("0").cast(pl.Int32).alias(f"secs_{prefix}"),
            pl.col(c).str.extract(r"-(\d+)$", 1).fill_null("0").cast(pl.Int32).alias(f"thir_{prefix}"),
        ])

    # Dog Grade & Whelped Date
    df = df.with_columns([
        pl.col("doggrade").replace({"M": "8", "": "9"}).cast(pl.Int16).alias("dgnum"),
        pl.col("whelped").str.to_datetime("%d %b %y", strict=False).alias("whelped_dt"),
    ])

    # Besttime logic
    df = df.with_columns([
        (pl.col("besttime") == "NBT").cast(pl.Int16).alias("NBT"),
        (pl.col("besttime") == "FSH").cast(pl.Int16).alias("FSH"),
        (pl.col("besttime") == "FSTD").cast(pl.Int16).alias("FSTD"),
        pl.col("besttime").str.contains(r"\.Q$").cast(pl.Int16).alias("BT_Q"),
        pl.col("besttime").str.contains(r"\.H$").cast(pl.Int16).alias("BT_H"),
        pl.col("besttime").str.replace_all(r"[Q|H|NBT|FSH|FSTD]", "").replace("", "0").cast(pl.Float32).alias("BT_NUM")
    ]).drop("besttime")

    # Weight
    df = df.with_columns(
        pl.when(pl.col("weight").is_null())
        .then(30.0)
        .otherwise(pl.col("weight").cast(pl.Float32))
        .alias("weight")
    )

    return df


def create_race_result_features(df: pl.DataFrame) -> pl.DataFrame:

    # Standardize place column. "=" means a tie
    df = df.with_columns(
        pl.col("place").str.replace("=", "").cast(pl.Int32, strict=False).fill_null(0).alias("placenum")
    )

    df = create_speedc(df)

    # PIR (Position in Running) Parsing
    df = df.with_columns([
        pl.col("pir").fill_null(""),
        pl.col("pir").str.splitn("/", 2).struct.field("field_0").alias("pir_prefix")
    ])

    df = df.with_columns([
        (pl.col("pir_prefix").str.to_uppercase() == "M").cast(pl.Float32).alias("pirM"),
        (pl.col("pir_prefix").str.to_uppercase() == "Q").cast(pl.Float32).alias("pirQ"),
        (pl.col("pir_prefix").str.to_uppercase() == "S").cast(pl.Float32).alias("pirS"),
        pl.col("pir").str.replace_all(r"[^0-9]", "").alias("pir_digits")
    ])

    # Extract digits for PIR positions
    for i in range(1, 6):
        df = df.with_columns(
            pl.col("pir_digits").str.slice(i-1, 1).cast(pl.Float32, strict=False).fill_null(0.0).alias(f"pir{i}")
        )
    return df


def validate_datasets(datasets: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """
    Validate datasets before saving.

    Args:
        datasets: Dictionary of dataset names to DataFrames

    Returns:
        Validated datasets (same dictionary)

    Raises:
        ValueError: If validation fails
    """
    for dataset_name, df in datasets.items():
        if not isinstance(df, pl.DataFrame):
            raise ValueError(f"Dataset {dataset_name} is not a Polars DataFrame")

        if df.height == 0:
            raise ValueError(f"Dataset {dataset_name} is empty")

        print(f"Validated {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")

    return datasets


def clean():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aws")

    print("Loading...")
    train = join_data(path)
    train = clean_track(train)
    train = create_index(train)

    print(f"Loaded unique races: {train.select('idx').n_unique()}")

    print("Performing missingness checks...")
    missingness_check(train)

    # Remove bad rows
    train = drop_no_box(train)
    train = drop_duplicates(train)

    # Clean and create features
    train = clean_race_features(train)
    train = create_dog_form_features(train)
    train = create_race_result_features(train)
    train = clean_margins(train)

    # Memory optimization
    train = train.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32)
    ])

    #Missingness check after cleaning
    print("Missingness after cleaning:")
    missingness_check(train)

    # Save
    os.makedirs("data/clean", exist_ok=True)
    train.write_parquet("data/clean/train.parquet")
    print("Done. Cleaned data saved to parquet.")
    
    # Check that today's races are in the file
    dt_str = (datetime.utcnow() + timedelta(hours=10)).strftime("%Y-%m-%d")
    todays_races = train.filter(pl.col("date") == dt_str)
    if todays_races.height > 0:
        print(f"Today's races ({dt_str}): {todays_races.select('idx').n_unique()} races found.")
    else:
        print(f"WARNING: No races found for today ({dt_str}).")

if __name__ == "__main__":
    clean()
