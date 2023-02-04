#%%
import re

import hvplot.pandas
import holoviews as hv
import panel as pn
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import (
# MinMaxScaler,
# OneHotEncoder,
# OrdinalEncoder,
# RobustScaler,
# StandardScaler,
# )

# hv.extension("bokeh")
pn.extension()

# %%

# Function to load arff files


def load_arff(fn: str, str2cat: bool = True) -> pl.DataFrame:
    """Load arff files into a polars dataframe.

    Args:
        fn (str): file name.
        str2cat (bool, optional): Cast columns with str datatype as categorical if True.
            Defaults to True.

    Returns:
        pl.DataFrame: polars dataframe
    """

    pattern = re.compile("@attribute .*")
    columns = list()

    with open(fn) as f:
        skiprows = 0
        line = ""

        while "@data" not in line:

            line = f.readline()
            match = re.findall(pattern, line)

            if match:
                columns.append(match[0].split(" ")[1])

            skiprows += 1

    df = pl.read_csv(fn, skip_rows=skiprows, has_header=False, ignore_errors=True)
    df.columns = columns

    # remove ' from strings
    df = df.with_columns(pl.col(pl.Utf8).str.replace_all("'", ""))

    # str to category
    if str2cat:
        df = df.with_column(pl.col(pl.Utf8).cast(pl.Categorical))

    # decrease memory usage
    df = df.with_columns(pl.all().shrink_dtype())

    return df


def one_hot_encode(df: pl.DataFrame, cat_cols: list) -> pl.DataFrame:

    df = df.with_columns(
        [
            pl.when(pl.col(col).cast(pl.Utf8) == str(cat))
            .then(1)
            .otherwise(0)
            .alias("_".join([col, str(cat)]))
            for col in cat_cols
            for cat in df[col].unique()
        ]
    ).drop(columns=cat_cols)

    return df


def ordinal_encode(df: pl.DataFrame, cat_cols: list) -> pl.DataFrame:
    for col in cat_cols:
        my_map = {cat: n for n, cat in enumerate(df[col].unique())}
        df = df.with_column(pl.col(col).apply(lambda x: my_map[x]))

    return df


def binarize(
    df: pl.DataFrame, bins: dict, return_col: str = "category"
) -> pl.DataFrame:
    for col in bins:
        df = df.with_columns(
            pl.cut(df[col], bins[col]).select(pl.col(return_col).alias(col)).to_series()
        )
    return df


def standard_scale(df: pl.DataFrame, num_cols: list) -> pl.DataFrame:

    df = df.with_columns(
        [(pl.col(col) - pl.col(col).mean()) / pl.col(col).std() for col in num_cols]
    )

    return df


def min_max_scale(
    df: pl.DataFrame, num_cols: str, feature_range: tuple | list = (0, 1)
) -> pl.DataFrame:

    min_, max_ = feature_range
    df = df.with_columns(
        [
            (
                (pl.col(col) - pl.col(col).min())
                / (pl.col(col).max() - pl.col(col).min())
            )
            * (max_ - min_)
            + min_
            for col in num_cols
        ]
    )
    return df


def hist_plot(df: pl.DataFrame, x: str, **kwargs) -> hv.Histogram | hv.Bars:

    if df[x].dtype == pl.Categorical or df[x].dtype == pl.Utf8:
        return (
            df.select(pl.col(x).value_counts())
            .unnest(x)
            .to_pandas()
            .sort_values(x)
            .hvplot.bar(x=x, y="counts", **kwargs)
        ).opts(title=f"counts per {x}")

    else:
        if not "bins" in kwargs:
            kwargs["bins"] = max(10, int(df[x].max() - df[x].min()))

        if not "bin_range" in kwargs:
            kwargs["bin_range"] = (df[x].min(), df[x].max())

    return df[x].to_pandas().hvplot.hist(**kwargs).opts(title=f"counts per {x}")


def line_plot(df: pl.DataFrame, x: str, **kwargs) -> hv.NdOverlay:

    df_ = (
        df.filter(pl.col("ClaimAmount") > 0)
        .groupby(x)
        .agg(
            [
                pl.col("ClaimAmount").mean().alias("mean"),
                (
                    pl.col("ClaimAmount").std() / pl.col("ClaimAmount").count().sqrt()
                ).alias("sem"),
            ]
        )
        .with_columns(
            [
                (pl.col("mean") + pl.col("sem")).alias("upper_band"),
                (pl.col("mean") - pl.col("sem")).alias("lower_band"),
            ]
        )
        .to_pandas()
        .sort_values(x)
    )
    return (
        df_.hvplot(x=x, y="mean")
        * df_.hvplot.scatter(x=x, y="mean")
        * df_.hvplot.area(x=x, y="lower_band", y2="upper_band", alpha=0.25, hover=False)
    ).opts(title=f"mean ClaimAmount per {x}")


def plot_corr(
    df: pl.DataFrame, num_cols: list, cat_cols: list, method: str = "pearson", **kwargs
) -> hv.HeatMap:
    return (
        df.pipe(ordinal_encode, cat_cols)
        .select(num_cols + cat_cols)
        .to_pandas()
        .corr(method=method)
        .hvplot.heatmap(title=f"{method} cross correlation", **kwargs)
    )


def plot(
    df: pl.DataFrame, x: str, hist_kwargs: dict = dict(), line_kwargs: dict = dict()
) -> hv.Layout:

    return (
        hist_plot(df=df, x=x, **hist_kwargs) + line_plot(df=df, x=x, **line_kwargs)
    ).cols(1)


# %%

# Load and join datasts

df_freq = load_arff("../data/raw/freMTPL2freq.arff", str2cat=False)
df_sev = load_arff("../data/raw/freMTPL2sev.arff", str2cat=False)

df_sev = df_sev.groupby("IDpol").sum()

df = (
    df_freq.join(df_sev, on="IDpol", how="left")
    .with_columns(
        [
            pl.col("ClaimAmount").fill_null(0),
        ]
    )
    .unique()
    .drop(["IDpol", "ClaimNb"])
)

# df.describe()

#%%

# plot Exposure and AvgClaimAmount


# df["Exposure"].to_pandas().hvplot.box() + df.filter(
#    pl.col("ClaimAmount") > 0
# ).select(pl.col("ClaimAmount")).to_pandas().hvplot.box(logy=True, grid=True)

df[["ClaimAmount", "Exposure"]].to_pandas().describe()

#%%

# filter data
df = df.filter(
    pl.col("ClaimAmount")
    < (pl.col("ClaimAmount").filter(pl.col("ClaimAmount") > 0)).quantile(0.99)
).with_columns(
    [
        pl.col("Exposure").clip(0, 1),
        (pl.col("Density").log()).alias("log(Density)"),
    ]
)


#%%
# plot hist and mean claimamount
columns = [
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "log(Density)",
    "Region",
]

selector = pn.widgets.Select(options=columns)
import numpy as np

pn.Row(
    selector,
    pn.bind(
        plot,
        df=df.pipe(
            binarize, {"log(Density)": np.arange(0, 12)}, return_col="break_point"
        ),
        x=selector,
    ),
).embed()

#%%

# features

cat_cols = ["VehGas", "VehBrand", "Area", "Region", "VehPower"]
num_cols = ["VehAge", "DrivAge", "log(Density)", "BonusMalus"]
# cols = ["BonusMalus"]



#%%

# Modeltraining

# 1. Linear Regression

from sklearn.linear_model import TweedieRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def modelling(
    df: pl.DataFrame,
    model,
    cat_cols: list,
    num_cols: list,
    min_claimamount:float|int=1e-9,
    one_hot_encode_: bool = True,
    ordinal_encode_: bool = False,
    standard_scale_: bool = True,
    min_max_scale_: bool = False,
    
):

    X = df.filter(pl.col("ClaimAmount") >= min_claimamount).select(
        cat_cols + num_cols
    )

    if one_hot_encode_:
        X = X.pipe(one_hot_encode, cat_cols)

    if ordinal_encode_:
        X = X.pipe(ordinal_encode, cat_cols)

    if standard_scale_:
        X = X.pipe(standard_scale, num_cols)

    if min_max_scale_:
        X = X.pipe(min_max_scale, num_cols)

    X = X.to_pandas()

    y = (
        df.filter(pl.col("ClaimAmount") >= min_claimamount)
        .select(
            [
                (pl.col("ClaimAmount") / pl.col("Exposure")).alias("YearlyClaimAmount"),
                pl.col("Exposure"),
            ]
        )
        .to_pandas()
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1234
    )

    model.fit(X=X_train, y=y_train["YearlyClaimAmount"], sample_weight=y_train["Exposure"])
    
    y_pred = model.predict(X=X_test)
    
    mse = mean_squared_error(y_pred, y_test["YearlyClaimAmount"], sample_weight=y_test["Exposure"])
    mae = mean_absolute_error(y_pred, y_test["YearlyClaimAmount"], sample_weight=y_test["Exposure"])
    
    y_pred_sum = y_pred.sum()
    y_test_sum = y_test["YearlyClaimAmount"].sum()
    
    res = pd.Series([mse, mae, y_pred_sum, y_test_sum], index=["mse", "mea", "pred. sum", "test sum"])
    
    if hasattr(model, "feature_importance_"):
        feature_importance = pd.Series(index=model.feature_names_in_, data=model.feature_importances_).sort_values().hvplot.barh(height=600)
    else:
        feature_importance=None
        
    
    return model, res, feature_importance
#%%

r = TweedieRegressor(
    alpha=1e-1,
    fit_intercept=True,
    power=1.2,
    max_iter=10000,
    solver="newton-cholesky",
)

r.fit(X_train, y_train["YearlyClaimAmount"], sample_weight=y_train["Exposure"])

s1 = pd.Series(pd.Series(r.predict(X_test)))
s2 = pd.Series(y_test["YearlyClaimAmount"])
#%%
(
    s1.hist(backend="holoviews", bins=100, logy=False, bin_range=(0.00, 1e4))
    * s2.hist(backend="holoviews", bins=100, logy=False, bin_range=(0.00, 1e4))
)


#%%
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)


X = (
    df_.filter(pl.col("ClaimAmount") >= 0)
    .select(cat_cols + scale_cols)
    .pipe(ordinal_encode, cat_cols)
    # .pipe(standard_scale, scale_cols)
    .to_pandas()
)

y = (
    df_.filter(pl.col("ClaimAmount") >= 0)
    .select(
        [
            (pl.col("ClaimAmount") / pl.col("Exposure")).alias("YearlyClaimAmount"),
            pl.col("Exposure"),
        ]
    )
    .to_pandas()
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, random_state=2306
)


et = RandomForestRegressor(
    criterion="friedman_mse", n_estimators=100
)  # loss="huber", learning_rate=1)#max_depth=50, loss="squared_error", learning_rate=1.25, l2_regularization=False,)

et.fit(X_train, y_train["YearlyClaimAmount"], sample_weight=y_train["Exposure"])

s1 = pd.Series(pd.Series(et.predict(X_test)))
s2 = pd.Series(y_test["YearlyClaimAmount"])

(
    s1.hist(backend="holoviews", bins=100, logy=False, bin_range=(0.00, 1e4), alpha=0.5)
    * s2.hist(
        backend="holoviews", bins=100, logy=False, bin_range=(0.00, 1e4), alpha=0.5
    )
)


# %%

from sklearn.neural_network import MLPRegressor

X = (
    df_.filter(pl.col("ClaimAmount") >= 0)
    .select(cat_cols + scale_cols)
    .pipe(one_hot_encode, cat_cols)
    .pipe(min_max_scale, scale_cols)
    .to_pandas()
)

y = (
    df_.filter(pl.col("ClaimAmount") >= 0)
    .select(
        [
            (pl.col("ClaimAmount") / pl.col("Exposure")).alias("YearlyClaimAmount"),
            pl.col("Exposure"),
        ]
    )
    .to_pandas()
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, random_state=2306
)

nn = MLPRegressor(hidden_layer_sizes=(20, 10))

nn.fit(X_train, y_train["YearlyClaimAmount"])

s1 = pd.Series(pd.Series(nn.predict(X_test)))
s2 = pd.Series(y_test["YearlyClaimAmount"])

(
    s1.hist(backend="holoviews", bins=100, logy=False, bin_range=(0.00, 1e4), alpha=0.5)
    * s2.hist(
        backend="holoviews", bins=100, logy=False, bin_range=(0.00, 1e4), alpha=0.5
    )
)


#%%
data.describe()
# %%


data = data.with_column(
    (pl.col("ClaimAmount") / pl.col("Exposure")).alias("ClaimAmntExp")
)

data = data.with_columns(
    [
        (pl.col("VehAge") // 5 * 5 + 5).alias("VehAge_u"),
        (pl.col("DrivAge") // 10 * 10 + 10).alias("DrivAge_u"),
    ]
)
# %%

data_pd = data.to_pandas()

data_pd.hvplot.hist(
    y="VehAge",
    by="VehBrand",
    bins=int(data_pd["VehAge"].quantile(0.95)),
    bin_range=[0, data_pd["VehAge"].quantile(0.95)],
    subplots=True,
    responsive=True,
    height=150,
    shared_axes=False,
    normed=False,
    cmap="category20c",
).cols(4)

#%%
data_pd.hvplot.hist(
    y="DrivAge",
    by="VehBrand",
    bins=20,
    bin_range=[0, 100],
    subplots=True,
    responsive=True,
    height=150,
    shared_axes=False,
    normed=False,
    cmap="category20c",
).cols(4)
# %%

data_pd.groupby(["DrivAge_u", "VehBrand"])[
    "ClaimAmount"
].describe().stack().reset_index().rename(
    {"level_1": "stat", 0: "value"}, axis=1
).hvplot.bar(
    x="DrivAge_u",
    color="DrivAge_u",
    by="stat",
    y="value",
    subplots=True,
    shared_axes=False,
    responsive=True,
    height=150,
    cmap="category20c",
).cols(
    4
)

# %%
(
    data_pd.query(f"ClaimAmount<{data_pd['ClaimAmount'].quantile(0.95)}")
    .hvplot.hist(
        by=["VehBrand"],
        y="ClaimAmount",
        subplots=True,
        responsive=True,
        height=150,
        normed=False,
        shared_axes=False,
    )
    .cols(4)
    + data_pd.query(f"ClaimAmount<{data_pd['ClaimAmount'].quantile(0.95)}")
    .hvplot.hist(
        by=["DrivAge_u"],
        y="ClaimAmount",
        subplots=True,
        responsive=True,
        height=150,
        normed=False,
        shared_axes=False,
    )
    .cols(4)
    + data_pd.query(
        f"ClaimAmount<{data_pd['ClaimAmount'].quantile(0.95)} & VehAge<{data_pd['VehAge'].quantile(0.95)}"
    )
    .hvplot.hist(
        by=["VehAge"],
        y="ClaimAmount",
        subplots=True,
        responsive=True,
        height=150,
        normed=False,
        shared_axes=False,
    )
    .cols(4)
).cols(1)
# %%
data_pd.describe()
# %%
data_pd.hvplot.hist(
    by=["VehPower"],
    y="ClaimAmntExp",
    subplots=True,
    responsive=True,
    height=200,
    normed=False,
    shared_axes=False,
    bins=50,
    bin_range=(0, 5000),
).cols(4)
# %%
data_pd["ClaimAmount"].quantile(0.95)
# %%

from sklearn.preprocessing import OneHotEncoder, StandardScaler

data_ohe = pl.DataFrame(
    OneHotEncoder(sparse_output=False).fit_transform(
        data.select(pl.col(["Area", "Region", "VehBrand", "VehGas"])).to_numpy()
    )
)
data_ohe.columns = []

data_normed = pl.DataFrame(
    StandardScaler().fit_transform(
        data.select(
            pl.col(["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"])
        ).to_numpy()
    )
)
data_normed.columns = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
data_normed
