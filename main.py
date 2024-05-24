from pandas import *
from pandas.plotting import scatter_matrix
from numpy import *
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot


if __name__ == '__main__':
    df = read_csv('CSV/laptop_price_data.csv')

    # df.hist()
    # pyplot.show()
    # scatter_matrix(df)

    df['Condition'] = df['Condition'].replace({'Very Good - Refurbished': 'Used'})
    df['Condition'] = df['Condition'].replace({'Open box': 'Used'})
    df['Condition'] = df['Condition'].replace({'Excellent - Refurbished': 'Used'})
    df['Condition'] = df['Condition'].replace({'Good - Refurbished': 'Used'})

    df['Screen_Size'] = df['Screen_Size'].replace({10.: 10.1})
    df['Screen_Size'] = df['Screen_Size'].replace({10.5: 10.1})
    df['Screen_Size'] = df['Screen_Size'].replace({11.6: 12.4})
    df['Screen_Size'] = df['Screen_Size'].replace({12.: 12.4})
    df['Screen_Size'] = df['Screen_Size'].replace({12.3: 12.4})
    df['Screen_Size'] = df['Screen_Size'].replace({12.5: 12.4})
    df['Screen_Size'] = df['Screen_Size'].replace({13.: 13.3})
    df['Screen_Size'] = df['Screen_Size'].replace({13.4: 13.3})
    df['Screen_Size'] = df['Screen_Size'].replace({13.5: 13.3})
    df['Screen_Size'] = df['Screen_Size'].replace({13.9: 14.1})
    df['Screen_Size'] = df['Screen_Size'].replace({14.: 14.1})
    df['Screen_Size'] = df['Screen_Size'].replace({14.4: 14.1})
    df['Screen_Size'] = df['Screen_Size'].replace({14.5: 14.1})
    df['Screen_Size'] = df['Screen_Size'].replace({15.: 15.6})
    df['Screen_Size'] = df['Screen_Size'].replace({15.3: 15.6})
    df['Screen_Size'] = df['Screen_Size'].replace({15.4: 15.6})
    df['Screen_Size'] = df['Screen_Size'].replace({15.5: 15.6})
    df['Screen_Size'] = df['Screen_Size'].replace({15.55: 15.6})
    df['Screen_Size'] = df['Screen_Size'].replace({16.: 15.6})
    df['Screen_Size'] = df['Screen_Size'].replace({16.1: 15.6})
    df['Screen_Size'] = df['Screen_Size'].replace({17: 17.3})
    df['Screen_Size'] = df['Screen_Size'].replace({18: 17.3})

    df['GPU_Type'] = df['GPU_Type'].replace({'Integrated/On-Board Graphics': 'Integrated',
                                             nan: 'Integrated',
                                             'Intel UHD Graphics': 'Integrated',
                                             'Intergrated Intel UHD Graphics': 'Integrated',
                                             'Intel Iris Xe Graphics': 'Integrated',
                                             'Iris Xe Graphics': 'Integrated',
                                             'Intel® UHD Graphics': 'Integrated'})


    list_of_sizes = []
    for line in df['Screen_Size']:
        if line not in list_of_sizes:
            list_of_sizes.append(line)

    X = df[['Screen_Size', 'GPU_Type', 'Condition']]  # Features
    y = df['Price']

    categorical_cols = ['GPU_Type', 'Condition']
    numerical_cols = ['Screen_Size']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)
    # print(metrics.accuracy_score(y, y_pred))

    print("\nTo estimate the price of your desired laptop, please answer the following questions: ")
    screen_size = 1.1
    while screen_size not in list_of_sizes:
        screen_size = float(input("\nWhat screen size are you looking for? (10.1, 12.4, 13.3, 14.1, 15.6, 17.3): "))

    gpu = 'hello'
    while gpu.lower() not in ['yes', 'no']:
        gpu = input("Do you need a dedicated GPU Please answer yes/no: ")

    condition = 'none'
    while condition.lower() not in ['new', 'used']:
        condition = input("Are you looking for a new or used laptop? ").title()

    if gpu.lower() == 'yes':
        gpu = 'Dedicated Graphics'
    elif gpu.lower() == 'no':
        gpu = 'Integrated'

    new_laptop_criteria = DataFrame(
        {'GPU_Type': [gpu], 'Condition': [condition], 'Screen_Size': [screen_size]})

    predicted_price = model.predict(new_laptop_criteria)
    formatted_price = "{:.2f}".format(float(predicted_price[0]))

    print(f"\nPredicted Price: ${formatted_price}")





