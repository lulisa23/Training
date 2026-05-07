"""Column definitions shared by training and prediction scripts."""

from __future__ import annotations

from dataclasses import dataclass


FEATURE_COLUMNS: tuple[str, ...] = (
    "rA",
    "rB",
    "VA",
    "VB",
    "Octahed",
    "EN_PA",
    "EN_PB",
    "EA",
    "EB",
    "EAA",
    "EAB",
    "DA",
    "DB",
    "IPA",
    "IPB",
    "NMA",
    "NMB",
    "A",
    "B",
    "MASS",
    "NUMBER",
    "TmA",
    "TmB",
    "Ts",
    "ts",
    "th",
)

TEMPERATURE_COLUMN = "temperature_c"

METADATA_COLUMNS: tuple[str, ...] = (
    "compound",
    "formula",
    "A_site",
    "B_site",
    "source",
    "notes",
)

TARGET_DEFAULT = "rho"


@dataclass(frozen=True)
class FeatureInfo:
    symbol: str
    category: str
    description_zh: str


FEATURE_DICTIONARY: tuple[FeatureInfo, ...] = (
    FeatureInfo("rA", "尺寸因素", "A 位离子半径"),
    FeatureInfo("rB", "尺寸因素", "B 位离子半径"),
    FeatureInfo("VA", "尺寸因素", "A 位原子体积"),
    FeatureInfo("VB", "尺寸因素", "B 位原子体积"),
    FeatureInfo("Octahed", "电化学因素", "八面体因子"),
    FeatureInfo("EN_PA", "电化学因素", "A 位鲍林电负性"),
    FeatureInfo("EN_PB", "电化学因素", "B 位鲍林电负性"),
    FeatureInfo("EA", "键合参数", "A 位第一电离能"),
    FeatureInfo("EB", "键合参数", "B 位第一电离能"),
    FeatureInfo("EAA", "键合参数", "A 位电子亲和性"),
    FeatureInfo("EAB", "键合参数", "B 位电子亲和性"),
    FeatureInfo("DA", "键合参数", "A 位离子位移"),
    FeatureInfo("DB", "键合参数", "B 位离子位移"),
    FeatureInfo("IPA", "键合参数", "A 位离子极化率"),
    FeatureInfo("IPB", "键合参数", "B 位离子极化率"),
    FeatureInfo("NMA", "键合参数", "A 位核磁矩"),
    FeatureInfo("NMB", "键合参数", "B 位核磁矩"),
    FeatureInfo("A", "原子因素", "A 位原子掺杂比例"),
    FeatureInfo("B", "原子因素", "B 位原子掺杂比例"),
    FeatureInfo("MASS", "原子因素", "掺杂原子相对质量"),
    FeatureInfo("NUMBER", "原子因素", "掺杂原子相对数目"),
    FeatureInfo("TmA", "物理因素", "A 位元素熔点"),
    FeatureInfo("TmB", "物理因素", "B 位元素熔点"),
    FeatureInfo("Ts", "工艺因素", "烧结温度"),
    FeatureInfo("ts", "工艺因素", "烧结时间"),
    FeatureInfo("th", "工艺因素", "保温时间"),
)


def required_feature_columns(include_temperature: bool) -> list[str]:
    """Return required feature columns for a model run."""
    columns = list(FEATURE_COLUMNS)
    if include_temperature:
        columns.append(TEMPERATURE_COLUMN)
    return columns


def missing_columns(columns: set[str], required: list[str]) -> list[str]:
    """Return required column names absent from an input table."""
    return [column for column in required if column not in columns]
