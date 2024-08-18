import os

def count_files_per_subfolder(directory):
    files_count = {}

    for root, dirs, files in os.walk(directory):
        # Calculate the number of files in the current directory
        num_files = len(files)
        # Get the relative path of the folder from the root directory
        relative_path = os.path.relpath(root, directory)
        # Store the count in the dictionary
        files_count[relative_path] = num_files

    return files_count

# Example usage
directory_path = '/Users/felix/Programming/llm-graph-based-problem-solving/examples/bigbench_hard/results/replicate-llama3-8b-ollama_io-io_zs-cot-cot_zeroshot-cot_sc-tot-plan_solve-plan_solve_plus-got_2024-08-11_19-25-53'
files_count = count_files_per_subfolder(directory_path)

# Print the number of files per subfolder
total =9 * 6511 + 27
done = 0
for folder, count in files_count.items():
    done += count
    if count != 250:
        print(f"Folder: {folder}, Number of files: {count}")

print(f"Total files: {done}/{total}. Percentage: {done/total*100:.2f}%")