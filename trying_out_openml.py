import openml
import pandas as pd

pd.set_option("display.max_rows", 200)  # so wide datasets don't print as "..."

dataset = openml.datasets.get_dataset(42769)
df, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)


def summarize(df):
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Duplicate rows:", df.duplicated().sum())
    print()

    summary = pd.DataFrame(
        {
            "Dtype": df.dtypes,
            "Missing": df.isnull().sum(),
            "Missing_%": (df.isnull().mean() * 100).round(1),
            "Unique": df.nunique(),
            "Most_common": df.mode().iloc[0],
            "Min": df.min(numeric_only=True),
            "Mean": df.mean(numeric_only=True).round(2),
            "Max": df.max(numeric_only=True),
        }
    )
    return summary


print(summarize(df))

print()
print("Target:", y.name)
print(y.value_counts())
