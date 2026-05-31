import pandas as pd
import unittest

from pipeline import append_result_row


class PipelineResultAppendTest(unittest.TestCase):
    def test_append_result_row_preserves_columns_and_appends_row(self):
        columns = ["True SPO", "Unamb SPO", "MSE", "Elapsed", "Epochs"]
        df = pd.DataFrame(columns=columns)
        row = {
            "True SPO": 0.1,
            "Unamb SPO": 0.2,
            "MSE": 0.3,
            "Elapsed": 1.5,
            "Epochs": 4,
        }

        result = append_result_row(df, row)

        self.assertEqual(list(result.columns), columns)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].to_dict(), row)


if __name__ == "__main__":
    unittest.main()
