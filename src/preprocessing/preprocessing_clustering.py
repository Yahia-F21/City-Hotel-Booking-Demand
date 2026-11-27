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


# ----------------------------------------------------------
# Clustering: no target, no ADR-derived columns, no split
# ----------------------------------------------------------
# The scaler / encoder / PCA are fitted on all rows (there is nothing to leak without a target).
X = df[BASE_NUM + CAT_COLS]
A = make_preprocessor(BASE_NUM).fit_transform(X)
pca = PCA(n_components=0.90, random_state=RANDOM_STATE).fit(A)
X_cluster = pd.DataFrame(pca.transform(A),
                         columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
                         index=X.index)

# ----------------------------------------------------------
# Output
# ----------------------------------------------------------
if __name__ == "__main__":
    print("rows after cleaning:", len(df))
    print("Clustering:", X_cluster.shape)

    X_cluster.to_csv(r"output data\Clustering data\Clustering data.csv", index=False)
   