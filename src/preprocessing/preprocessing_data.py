import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", message="Found unknown categories")

RANDOM_STATE = 42
DROP_DUPLICATES = True   # your current choice; True removes ~1/3 of the rows (exact duplicates)
CLUSTER_PCA = True        # reduce the encoded clustering data with PCA (fitted on train only)

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
df["EstTotalPrice"] = df["ADR"] * df["TotalNights"]
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


def prepare(preprocessor, X_tr, X_te, pca=False):
    """Learn scaling/encoding from the TRAIN rows only, apply to train and test.
    This only prepares the data; no model is trained."""
    A = preprocessor.fit_transform(X_tr)
    B = preprocessor.transform(X_te)
    cols = list(preprocessor.get_feature_names_out())
    if pca:
        p = PCA(n_components=0.90, random_state=RANDOM_STATE).fit(A)
        A, B = p.transform(A), p.transform(B)
        cols = [f"PC{i + 1}" for i in range(A.shape[1])]
    return (pd.DataFrame(A, columns=cols, index=X_tr.index),
            pd.DataFrame(B, columns=cols, index=X_te.index))


# ----------------------------------------------------------
# Classification: IsCanceled
# ----------------------------------------------------------
# ADR and EstTotalPrice are known at booking time, so they are valid features here
cls_num = BASE_NUM + ["ADR", "EstTotalPrice"]
X = df[cls_num + CAT_COLS]
y = df["IsCanceled"]
X_tr, X_te, y_class_tr, y_class_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_class_tr, X_class_te = prepare(make_preprocessor(cls_num), X_tr, X_te)

# ----------------------------------------------------------
# Regression: ADR
# ----------------------------------------------------------
# ADR == 0 rows are mostly complimentary stays; no ADR / EstTotalPrice in the features.
# IsCanceled is kept as a feature here (remove it if the model must work at booking time).
df_reg = df[df["ADR"] > 0]
reg_num = BASE_NUM + ["IsCanceled"]
X = df_reg[reg_num + CAT_COLS]
y = df_reg["ADR"]
X_tr, X_te, y_reg_tr, y_reg_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)
X_reg_tr, X_reg_te = prepare(make_preprocessor(reg_num), X_tr, X_te)

# ----------------------------------------------------------
# Clustering: no target, no ADR-derived columns
# ----------------------------------------------------------
X = df[BASE_NUM + CAT_COLS]
X_tr, X_te = train_test_split(X, test_size=0.2, random_state=RANDOM_STATE)
X_cluster_tr, X_cluster_te = prepare(make_preprocessor(BASE_NUM), X_tr, X_te, pca=CLUSTER_PCA)

# ----------------------------------------------------------
# Output: only the train / test data
# ----------------------------------------------------------
if __name__ == "__main__":
    print("rows after cleaning:", len(df))
    print("Classification:", X_class_tr.shape, X_class_te.shape,
          y_class_tr.shape, y_class_te.shape)
    print("Regression:    ", X_reg_tr.shape, X_reg_te.shape,
          y_reg_tr.shape, y_reg_te.shape)
    print("Clustering:    ", X_cluster_tr.shape, X_cluster_te.shape)
    
    # X_class_tr.to_csv(r"output data unduplicated\Classification data\Classification train data.csv", index=False)
    # y_class_tr.to_csv(r"output data unduplicated\Classification data\Classification train result.csv", index=False)

    # X_class_te.to_csv(r"output data unduplicated\Classification data\Classification test data.csv", index=False)
    # y_class_te.to_csv(r"output data unduplicated\Classification data\Classification test result.csv", index=False)


    # X_reg_tr.to_csv(r"output data unduplicated\Regression data\Regression train data.csv", index=False)
    # y_reg_tr.to_csv(r"output data unduplicated\Regression data\Regression train result.csv", index=False)

    # X_reg_te.to_csv(r"output data unduplicated\Regression data\Regression test data.csv", index=False)
    # y_reg_te.to_csv(r"output data unduplicated\Regression data\Regression test result.csv", index=False)

    
    # X_cluster_tr.to_csv(r"output data unduplicated\Clustering data\Clustering train data.csv", index=False)
    # X_cluster_te.to_csv(r"output data unduplicated\Clustering data\Clustering test data.csv", index=False)