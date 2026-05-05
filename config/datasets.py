"""Dataset configurations for greys-data."""

from dataclasses import dataclass


FORM_BASE_QUERY = """
select
  -- Race Level Features
  lower(trim(regexp_replace( fp.track ,'^The|\\(.+\\)| ',''))) as track,
  date_parse( racetimedateutc, '%d %b %y %I:%i%p') as datetime_utc,
  date_format(date_parse( racetimedateutc, '%d %b %y %I:%i%p'),'%Y-%m-%d') as date_utc,
  try_cast( fp.racenum as bigint) as racenum,
  fp.racename,
  fp.racetime,
  fp.racetimedateutc,
  fp.distance,
  fp.racegrade,
  fp.prizemoney1,
  fp.prizemoney2,
  fp.prizemoney3,
  fp.prizemoney4,
  fp.prizemoney5,
  fp.prizemoney6,
  fp.prizemoney7,
  fp.prizemoney8,
  fp.quali,
  fp.hurdle,
  fp.handicap,
  fp.gradecode,
  fp.racecomment,
  b.event_id,

  -- Dog-Level Features
  b.selection_id,
  replace(regexp_replace(trim(upper( fp.dogname )),'RES\\.$|[^a-zA-Z0-9 ]|\\s+|\\.|\\?',''),'''','') as dogname,
  fp.racebox,
  fp.dogname as fp_dogname,
  fp.besttime,
  fp.doghandicap,
  fp.rating,
  fp.speed,
  fp.dogcomment,
  fp.startstot,
  fp.startsttd,
  fp.trainer,
  fp.suburb,
  fp.owner,
  fp.sire,
  fp.dam,
  fp.colour,
  fp.sex,
  fp.whelped,
  fp.doggrade,
  fp.dogprize,
  fp.agedprizemoney,
  b.win_bsp,
  b.win_target,
  b.place_bsp,
  b.place_target

from doger.fullplus fp

left join doger.bsp_place_view b
  on date_format(date_parse( racetimedateutc, '%d %b %y %I:%i%p'), '%Y-%m-%d') = date_format(b.datetime_utc, '%Y-%m-%d')
  and replace(regexp_replace(trim(upper( fp.dogname )),'RES\\.$|[^a-zA-Z0-9 ]|\\s+|\\.|\\?',''),'''','') = b.dogname
  and try_cast( fp.racenum as bigint) = try_cast(replace(split_part(b.event_name,' ',1),'R','') as bigint)
"""

RESULTS_BASE_QUERY = """
select
 -- Same per runner
datetime_local
, timezoneid
, timezonename
, track
, racenum
, distance
, racegrade
, racename
, wintime
, split1time
, split1rug
, split2time
, split2rug
, split3time
, split3rug
, windiv
, place1div
, place2div
, place3div
, quin
, exacta
, trifecta
, pick4
, rd
, dd
, quad

-- Per Runner Features
, dogname
, place
, trainer
, box
, rug
, weight
, startprice
, margin1
, margin2
, pir
, checks
, comments
, splitmargin
, runtime
, prizemoney
from doger.results_view
"""


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str
    query: str
    output_file: str
    description: str = ""


DATASETS = {
    "form": DatasetConfig(
        name="form",
        query=FORM_BASE_QUERY,
        output_file="dataset_form.parquet",
        description="Form data query from Athena",
    ),
    "results": DatasetConfig(
        name="results",
        query=RESULTS_BASE_QUERY,
        output_file="dataset_results.parquet",
        description="Results data query from Athena",
    ),
}


def get_dataset(name: str) -> DatasetConfig:
    """Get dataset configuration by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name]


__all__ = ["DatasetConfig", "DATASETS", "get_dataset"]
