import argparse
import csv
import os

# TODO: Module docstring: please
# TODO: Rename this camelcase garbage

def calculate_edit_distance(s1, s2):
    # Initialize a 2D array to store edit distances
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    # Fill the first row and column
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    # Calculate edit distance
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[-1][-1]

def all_na(row):
    return all(value == "NA" for value in row.values())

# This removes columns in labels that are not present in predictions
def remove_extra_columns(predictions, labels):
    with open(predictions, 'r') as file1, open(labels, 'r') as file2:
        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)
        headers1 = next(reader1)
        headers2 = next(reader2)

    columns_to_remove = [header for header in headers2 if header not in headers1]

    with open(labels, 'r') as file:
        reader = csv.DictReader(file)
        data2 = [{header: row[header] for header in headers2 if header not in columns_to_remove} for row in reader]

    # TODO: reorder columns in labels
    return data2

def remove_missing_rows(dataset_folder, labels):
    def comparator(file_string):
        try:
            return int(file_string[17:-4])
        except:
            # arbitrary large number to kick weird files to the end.
            return 100000000

    files = os.listdir(dataset_folder)
    new_labels = []
    for file_name in sorted(files, key=comparator):
        try:
            new_labels.append(labels[int(file_name[17:-4])-1])
        except:
            pass
    return new_labels

def add_filename_column(labels):
    for index in range(len(labels)):
        labels[index]["check_file_num"] = f"{index+1}"
    return labels


def calculate_average_edit_distance(dataset_folder, predictions, labels, verbose):
    skip_idxs = []
    with open(predictions, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)

    with open(predictions, 'r') as file:
        reader = csv.DictReader(file)
        predictionData = [{header: row[header] for header in headers} for row in reader]

    labelData = remove_extra_columns(predictions, labels)

    labelData = add_filename_column(labelData)

    # skip rows of files not present in dataset. This only applies to partial datasets
    labelData = remove_missing_rows(dataset_folder, labelData)

    avg_edit_distance = {header: 0 for header in headers}
    counts = {header: 0 for header in headers}
    accuracy = {header: 0 for header in headers}
    missing_reads = {header: 0 for header in headers}
    hit_rate = {header: 0 for header in headers}

    total_rows = 0
    for index, (row1, row2) in enumerate(zip(predictionData, labelData)):

        # Uncomment this if the dataset you use has lots of blank / "Record Of Deposit" images.
        # However, some true negatives may be skipped by this snippet.
        # if all_na(row1):
        #     skip_idxs.append(index)
        #     continue

        total_rows +=1

        for header in headers:
            value1 = row1[header]
            value2 = row2[header]

            if value1.upper() == "NA":
                missing_reads[header] += 1
                if verbose:
                    print("MICR process error!")
                    print("Check file num: " + row2["check_file_num"])
                    print()
                continue

            # data cleaning
            value1 = value1.replace(" ", "")
            if header == "Check Amount":
                for char in CHARS_TO_REMOVE:
                    value1 = value1.replace(char, "")
                value1 = str(float(value1))
                value2 = str(float(value2))
            else:
                value1 = value1.upper()
                value2 = value2.upper()


            # compute edit distance
            counts[header]+=1
            edit_dist = calculate_edit_distance(value1, value2)
            avg_edit_distance[header] += edit_dist

            if calculate_edit_distance(value1, value2) == 0:
                accuracy[header]+=1

            # Print Errors
            if verbose and calculate_edit_distance(value1, value2) > 0:# and header == "Check Amount":
                print("Prediction, Label")
                print(value1, value2)
                print("Check file num: " + row2["check_file_num"])
                print()

    # Calculate the average edit distance for each column
    for header in headers:
        avg_edit_distance[header] /= counts[header]
        accuracy[header] /= counts[header]
        hit_rate[header] = counts[header] / (counts[header] + missing_reads[header])

    return avg_edit_distance, accuracy, hit_rate, missing_reads, total_rows, skip_idxs


if __name__ == "__main__":
    # Must be predictions, then labels; labels may be a superset of predictions.
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='TODO')
    parser.add_argument('predictions', type=str, help='TODO')
    parser.add_argument('labels', type=str, help='TODO')
    parser.add_argument('--verbose', action='store_true', help='Set the flag to True')

    args = parser.parse_args()
    #LABELS = "mcd-test-3-labels.csv"
    #PREDICTIONS = "LLaVA_Benchmark-v1.6.csv"
    # characters to remove from check amount
    CHARS_TO_REMOVE = "$,"
    avg_edit_distance, accuracy, hit_rate, missing_reads, total_rows, skip_idxs = \
        calculate_average_edit_distance(args.dataset_folder, args.predictions, args.labels, args.verbose)
    print("Note: Edit distance and accuracy only account for hits.")
    print("Average Edit Distance for each column:")
    for header, distance in avg_edit_distance.items():
        print(f"{header}: {distance}")

    print("Accuracy for each column:")
    for header, acc in accuracy.items():
        print(f"{header}: {acc}")

    print("Hit Rate for each column:")
    for header, hit_rate in hit_rate.items():
        print(f"{header}: {hit_rate}")
