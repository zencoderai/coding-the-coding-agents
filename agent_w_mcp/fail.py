import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0, 100, size=(4, 4)), columns=["col1", "col2", "col3", "col4"])
print(df.shape)
