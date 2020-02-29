import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def gather_data_from_huggingface(root_dir: str, output_dir: str):
    all_models = {}
    for result_dir in os.listdir(root_dir):
        model_name = result_dir.split("/")[-1]
        model_results = []
        checkpoint_folder = None
        for checkpoint_folder in glob.glob(os.path.join(root_dir, result_dir, "*checkpoint-*/")):
            if not os.path.isfile(os.path.join(checkpoint_folder, "eval_results.txt")):
                print("No results file in path:", checkpoint_folder)
            else:
                results_on_check = {}
                with open(os.path.join(checkpoint_folder, "eval_results.txt"), "r") as f:
                    for index, line in enumerate(f):
                        name, value = line.split("=")
                        name, value = name.strip(), float(value.strip())
                        results_on_check[name] = value
            results_on_check["model"] = model_name
            results_on_check["checkpoint"] = checkpoint_folder.split("/")[-2]
            model_results.append(results_on_check)
        results_df = pd.DataFrame(model_results)
        if checkpoint_folder is not None:
            results_df.to_csv(os.path.join(checkpoint_folder, "model_results.csv"))
            ax = sns.lineplot(x="index", y="rmse", data=results_df.reset_index()) 
            plt.savefig(os.path.join(root_dir, result_dir, "rmse_plot.png"))
            plt.close()
            all_models[model_name] = results_df
            print("Wrote model file to ", os.path.join(result_dir, "rmse_plot.png"))

    if all_models:
        # now that all the results have been gathered, let's combine them
        full_df = pd.concat(list(all_models.values())).reset_index()
        full_df.to_csv(output_dir)
        # focus on lowest RMSE (could focus on Pearson, Spearmanr etc.)
        min_rmse = full_df.groupby("model")["rmse"].idxmin()
        best_results = full_df.iloc[min_rmse, :]

        # plot line plot here
        ax = sns.barplot(x="model", y="rmse", data=best_results) 
        plt.savefig(os.path.join(root_dir, "rmse_plot.png"))
        print("Wrote full file to ", os.path.join(root_dir, "rmse_plot.png"))
        plt.close()
        print(best_results)


if __name__ == "__main__":
    gather_data_from_huggingface("output/large", "regression_results_large.csv")
    gather_data_from_huggingface("output/base/", "regression_results_base.csv")