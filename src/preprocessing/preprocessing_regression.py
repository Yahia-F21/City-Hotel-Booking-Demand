import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", message="Found unknown categories")

RANDOM_STATE = 42
DROP_DUPLICATES = True   # True removes ~1/3 of the rows (exact duplicates)
VAL_SIZE = 0.15
TEST_SIZE = 0.15

df = pd.read_csv("H2.csv")

# ----------------------------------------------------------
# Cleaning
# ----------------------------------------------------------
# strip padding from every text column (e.g. "HB      ", "       NULL")
for c in df.columns:
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].str.strip()

df["Company"] = df["Company"].transform(lambda x: np.nan if x == "NULL" else x)
df["Agent"] = df["Agent"].transform(lambda x: np.nan if x == "NULL" else x)

df["Children"] = df["Children"].fillna(0)
df["Country"] = df["Country"].fillna("unknown")

if DROP_DUPLICATES:
    df = df.drop_duplicates()

df = df[df["ADR"] < 1000]                                             # extreme outlier (5400)
df = df[(df["Adults"] + df["Children"] + df["Babies"]) > 0]           # zero guests
df = df[(df["StaysInWeekendNights"] + df["StaysInWeekNights"]) > 0]   # zero nights
df = df.copy().reset_index(drop=True)

# ----------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------
df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]
df["HasKids"] = ((df["Children"] + df["Babies"]) > 0).astype(int)
df["RoomChanged"] = (df["ReservedRoomType"] != df["AssignedRoomType"]).astype(int)
df["HasAgent"] = df["Agent"].notna().astype(int)
df["HasCompany"] = df["Company"].notna().astype(int)
df["HasPrevCancellation"] = (df["PreviousCancellations"] > 0).astype(int)

# missing agent becomes its own category (must come AFTER HasAgent is computed)
df["Agent"] = df["Agent"].fillna("none").astype(str)

# month as a cycle, so December is next to January
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
month_num = df["ArrivalDateMonth"].map({m: i + 1 for i, m in enumerate(MONTHS)})
df["MonthSin"] = np.sin(2 * np.pi * month_num / 12)
df["MonthCos"] = np.cos(2 * np.pi * month_num / 12)

# tame the long right tails
df["Babies"] = df["Babies"].clip(upper=2)
for c in ["LeadTime", "PreviousCancellations", "PreviousBookingsNotCanceled",
          "BookingChanges", "DaysInWaitingList"]:
    df[c] = np.log1p(df[c])

# ----------------------------------------------------------
# Feature lists
# ----------------------------------------------------------
BASE_NUM = ["LeadTime", "ArrivalDateYear", "MonthSin", "MonthCos",
            "TotalNights", "TotalGuests", "HasKids", "IsRepeatedGuest",
            "PreviousCancellations", "PreviousBookingsNotCanceled",
            "HasPrevCancellation", "RequiredCarParkingSpaces",
            "TotalOfSpecialRequests", "HasAgent", "HasCompany"]

CAT_COLS = ["Meal", "MarketSegment", "DistributionChannel", "DepositType",
            "CustomerType", "ReservedRoomType", "Country", "Agent"]


def make_preprocessor(num_cols, cat_cols=CAT_COLS):
    """Scales numeric columns and one-hot encodes categorical ones."""
    return ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(
             handle_unknown="infrequent_if_exist",  # unseen levels -> "infrequent" bucket
             min_frequency=0.01,                    # levels under 1% of train rows -> bucket
             drop="if_binary",
             sparse_output=False), cat_cols)],
        verbose_feature_names_out=False)



def split_3way(X, y=None, stratify=False):
    """70% train / 15% validation / 15% test (optionally stratified on y)."""
    strat = y if stratify else None
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=VAL_SIZE + TEST_SIZE, stratify=strat, random_state=RANDOM_STATE)
    strat = y_tmp if stratify else None
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        stratify=strat, random_state=RANDOM_STATE)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def prepare(preprocessor, X_tr, X_va, X_te):
    """Learn scaling/encoding from the TRAIN rows only, apply to train, validation and test.
    This only prepares the data; no model is trained."""
    A = preprocessor.fit_transform(X_tr)
    B = preprocessor.transform(X_va)
    C = preprocessor.transform(X_te)
    cols = list(preprocessor.get_feature_names_out())
    return (pd.DataFrame(A, columns=cols, index=X_tr.index),
            pd.DataFrame(B, columns=cols, index=X_va.index),
            pd.DataFrame(C, columns=cols, index=X_te.index))


# ----------------------------------------------------------
# Regression: ADR
# ----------------------------------------------------------
# Timing: only information known at booking time is used.
# No ADR / EstTotalPrice (ADR-derived) and no IsCanceled (known only after the booking outcome).
# ADR == 0 rows are mostly complimentary stays, so they are removed.
df_reg = df[df["ADR"] > 0]
X = df_reg[BASE_NUM + CAT_COLS]
y = df_reg["ADR"]
X_tr, X_va, X_te, y_reg_tr, y_reg_va, y_reg_te = split_3way(X, y)
X_reg_tr, X_reg_va, X_reg_te = prepare(make_preprocessor(BASE_NUM), X_tr, X_va, X_te)

# ----------------------------------------------------------
# Output: only the train / validation / test data
# ----------------------------------------------------------
if __name__ == "__main__":
    print("rows after cleaning:", len(df), "| rows used for regression:", len(df_reg))
    print("Regression X (train/val/test):", X_reg_tr.shape, X_reg_va.shape, X_reg_te.shape)
    print("Regression y (train/val/test):", y_reg_tr.shape, y_reg_va.shape, y_reg_te.shape)

    X_reg_tr.to_csv(r"output data\Regression data\Regression train data.csv", index=False)
    y_reg_tr.to_csv(r"output data\Regression data\Regression train result.csv", index=False)

    X_reg_va.to_csv(r"output data\Regression data\Regression validation data.csv", index=False)
    y_reg_va.to_csv(r"output data\Regression data\Regression validation result.csv", index=False)
    

    X_reg_te.to_csv(r"output data\Regression data\Regression test data.csv", index=False)
    y_reg_te.to_csv(r"output data\Regression data\Regression test result.csv", index=False)
