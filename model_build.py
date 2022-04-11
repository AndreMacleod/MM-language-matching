# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read holdings as csv
mbie_holdings = pd.read_csv('mbie_fund_holdings.json')
print(mbie_holdings)

# Create a list of tuples, with holding name and corresponding holdings dataframe
mbie_holdings_list = []
for index, row in mbie_holdings.iterrows():
    json_str = row['holdings']
    json_str = json_str.replace("'", '"')
    mbie_holding_df = pd.read_json(json_str)
    name = row['fund_name']
    mbie_holdings_list.append((name,mbie_holding_df))

mbie_holdings_list[0][1]

# Load fossil fuel companies as both df and list objects
# CSV
ff_companies = pd.read_csv('ff_comps.csv', names=['holding'])
ff_companies
# FF companies list object
ff_companies_list = ff_companies.holding.tolist()
ff_companies_list

# Inital test to see how many match exactly
for fund in mbie_holdings_list:
    df = pd.merge(fund[1], ff_companies, on = 'holding', how='inner')
    print("Holding: " + fund[0])
    print("--------------------")
    print(df)
    print("--------------------")


# For similarity score, we use Levenstein distance
def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )

    return (matrix[size_x - 1, size_y - 1])

levenshtein("Adani Power Limited", "Adani Power Ltd")

# function to find smallest Leveshtein score for a company compared to all in holding
def smallest_levenshtein(holding_company):
    lev_score_list = []

    for ff_company in ff_companies_list:

        lev_score = levenshtein(holding_company, ff_company)
        lev_score_list.append(lev_score)

    min_lev_score = min(lev_score_list)
    min_lev_index = np.argmin(lev_score_list)
    company_levenshtein_score = min_lev_score
    corresponding_ff_company = ff_companies_list[min_lev_index]   

    return company_levenshtein_score, corresponding_ff_company


# Calculate Leveshtein score for similarity between each company in investemt holding and fossil fuels company
# We also have a threshold variable, above which we do not consider the company name to be a match - we adjust this based on testing, like a hyperparameter

def similarity_column(df, threshold):

    df['lev_score'] = None
    df['ff_company'] = ' '

    for i in range(0, len(df)):
        company_levenshtein_score, corresponding_ff_company = smallest_levenshtein(df.at[i, 'holding'])
        if company_levenshtein_score < threshold:
            df.at[i, 'lev_score'] = company_levenshtein_score
            df.at[i, 'ff_company'] = corresponding_ff_company

# Add similarity column for each fund
def fund_list_similarity(holding_list, threshold):
    for fund in holding_list:
        similarity_column(fund[1], threshold)
    return holding_list

fund_list_similarity(mbie_holdings_list, 6)

## Testing model

# Load macthed companies list
ff_names_matched = pd.read_csv('ff_names_matched.csv')

# Calculate accuracy of model using matched companies list
# We also return incorrect rows so user can see which are false positives and which are false negatives
def test(df):
    df = pd.merge(df, ff_names_matched, left_on = 'holding', right_on = 'company_name_mbie', how = 'left')
    df.fillna(' ', inplace = True)
    incorrect_rows = df.loc[df['ff_company'] != df['company_name_sustainalytics']]
    accuracy = 1 - (len(incorrect_rows) / len(df))
    return incorrect_rows, accuracy

# Calculate overall accuracy of funds by averaging separate accuracies of each fund
def find_accuracy(holdings_list):
    avg_accuracy = 0
    for fund in holdings_list:
        incorrect_rows, accuracy = test(fund[1])
        avg_accuracy += accuracy
    avg_accuracy = avg_accuracy/len(mbie_holdings_list)
    return avg_accuracy

# Testing with threshold = 8
mbie_holdings_list = fund_list_similarity(mbie_holdings_list, 8)
find_accuracy(mbie_holdings_list)

# Find best threshold
thresholds = list(range(1,10))
accuracies = []

# Find accuracy for each threshold value
for i in range(1,10):
    mbie_holdings_list = fund_list_similarity(mbie_holdings_list, i)
    accuracy = find_accuracy(mbie_holdings_list)
    accuracies.append(accuracy)
thresholds
accuracies

# Plot accuracy against threshold
plt.figure(figsize=(12,12))
plt.plot(thresholds, accuracies)
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.title("Accuracy against threshold")

plt.savefig('AccuracyThresholdPlot.png')
plt.show()

# We can see 6 and 7 are best threshold values. Let us choose threshold = 6 to run model
threshold = 6
mbie_holdings_list = fund_list_similarity(mbie_holdings_list, threshold)
find_accuracy(mbie_holdings_list)

# Calculate percentage of fossil fuel holdings each fund has
def calculate_ff_holdings(df):
    ff_rows = df.loc[df['ff_company'] != ' ']
    ff_holdings = ff_rows['amount'].sum()
    total_holdings = df['amount'].sum()
    #print(ff_holdings)
    #print(total_holdings)
    percentage_ff_holdings = ff_holdings / total_holdings * 100
    return percentage_ff_holdings

percentages = []
names = []
for fund in mbie_holdings_list:
    fund_percentage = calculate_ff_holdings(fund[1])
    fund_name = fund[0]
    print("Percentage of fossil fuel holdings in " + fund_name + ": " + str(fund_percentage) + "%")
    percentages.append(fund_percentage)
    names.append(fund_name)
percentages
names

# Plot percentage of each fund
plt.figure(figsize=(12,12))
plt.bar(names, percentages)
plt.xlabel("Fund Name")
plt.ylabel("Percentage of holdings in fossil fuels")
plt.title("Percentage of fossil fuel holdings in MBIE investment funds")
plt.savefig('FossilFuelInvestmentHoldings.png')

plt.show()

