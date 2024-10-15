import tkinter as tk
from tkinter import ttk
import warnings
from RUN import *
from Utils import utilities as ut
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import pandas as pd

Xsource = None
Lsource = None
source_w = None
Xtarget = None
Ltarget = None




def Run(Xsource, Ysource, source_w, Xtarget, Ytarget):
    AUC = []
    F = []
    Accuracy = []
    algorithms=["Boost","NB"]
    dup= np.array([0,0,0])
    scores=np.zeros(dup.shape[0])

    model = RUN(Xsource, Ysource, source_w, Xtarget, Ytarget, clf="NB")
    model.fit()

    for algo in algorithms:

        for i in range(0,20):
            # Load saved model if available
            model_filename = f"{algo}_model.joblib"
            try:
                model = joblib.load(model_filename)
            except FileNotFoundError:
                model = RUN(Xsource, Ysource, source_w, Xtarget, Ytarget, clf=algo)
                model.fit()
                # Save the model for future use
                joblib.dump(model, model_filename)

            model.predict()
            # print(model.testX)



            
            AUC.append(model.AUC)
            F.append(model.F)
            Accuracy.append(model.Accuracy)
        print(algo,": ","AUC: ",np.mean(AUC)," F1-Score: ",np.mean(F) ,"Accuracy: ",np.mean(Accuracy))
        values_array = np.array([np.mean(AUC), np.mean(F), np.mean(Accuracy)])
        scores = np.vstack((scores,values_array))


    auc_label.config(text=f"AUC Score: {AUC[0]:.4f}")
    f1_label.config(text=f"F1 Score: {F[0]:.4f}")
    accuracy_label.config(text=f"Accuracy: {Accuracy[0]:.4f}")

    print(scores)
    #result_label.config(text="AUC: {:.4f}  F1-Score: {:.4f}  Accuracy: {:.4f}".format(np.mean(AUC), np.mean(F),np.mean(Accuracy)))
    plot_results(scores)

def plot_results(results):
    num_algorithms = 3  # Number of algorithms
    num_scores = 3  # Number of scores (AUC, F1, Accuracy)
    print(results[1][0])

    max_score = max(max(result) for result in results[1:])

    # Create a grid of subplots
    fig, axes = plt.subplots(1,num_algorithms-1, figsize=(8, 6))
    fig.tight_layout(pad=4)  # Adjust the spacing between subplots
    score_index=0
    algorithms_list=["Boost","NB"]
    for algorithm_index in range(1,num_algorithms):
        
            # Create a bar plot for each algorithm and score
        ax = axes[score_index]
        bars = ax.bar(["AUC", "F1-Score", "Accuracy"], [results[algorithm_index][0], results[algorithm_index][1], results[algorithm_index][2]])
        ax.set_ylabel("Score")
        ax.set_title("Algorithm: "+ algorithms_list[algorithm_index-1])
        score_index=score_index+1
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")

        for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        ax.set_ylim(0, max_score + 0.1)
            # Set the y-axis format to show three decimal places
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.3f}".format(x)))

    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0)

    # Add a caption
    caption_label = ttk.Label(right_frame, text="Performance metrics for different algorithms.",font=("Arial",14))
    caption_label.grid(row=1, column=0)


def select_dataset():
    dataset_name = dataset_combobox.get()

    directory = "data/PROMISE/"

    if dataset_name:
        global Xsource, Lsource, source_w, Xtarget, Ltarget  # Declare as global variables
        Xtarget, Ltarget = ut.read_dataset(directory=directory,dataset_name=dataset_name)  # Replace with the appropriate path
        Ltarget[Ltarget > 1] = 1
        Xtarget = Xtarget

        Xsource = np.zeros((1, Xtarget.shape[1]))
        Lsource = [0]
        source_w = [0]
        sn = 0

        Xt_mean = Xtarget.mean(axis=0)

        dists = []
        Xs_mean = 0
        for dss in datasets:
            if dss != dataset_name:
                sn += 1
                X, L = ut.read_dataset(f"data/PROMISE/{dss}.csv")  # Replace with the appropriate path
                Xs_mean += X.mean(axis=0)

                # Check if X contains NaN values
                if np.isnan(X).any():
                    print(f"Warning: Dataset {dss} contains NaN values. Skipping.")
                else:
                    Xsource = np.concatenate((Xsource, X), axis=0)
                    Lsource = np.concatenate((Lsource, L), axis=0)

        Xsource = Xsource[1:, :]
        Lsource = Lsource[1:]
        source_w = source_w[1:]
        Lsource[Lsource > 1] = 1

        if np.isnan(Xsource).any():
            print("Error: Xsource contains NaN values.")
        else:
            try:
                Run(Xsource, Lsource, source_w, Xtarget, Ltarget)
            except Exception as e:
                print(f"Error in Run: {e}")

# def display_metrics():
#     # Generate and plot performance metrics for various algorithms
#     # Replace this with your actual code to generate metrics for different algorithms
#     # For simplicity, we use random data in this example
#     algorithms = ["Algorithm 1", "Algorithm 2", "Algorithm 3"]
#     scores = {"AUC": np.random.rand(3), "F1-Score": np.random.rand(3), "Accuracy": np.random.rand(3)}

#     fig, axes = plt.subplots(1, len(algorithms), figsize=(15, 5))
#     for i, algorithm in enumerate(algorithms):
#         ax = axes[i]
#         ax.bar(scores.keys(), scores[algorithm])
#         ax.set_title(algorithm)
#         ax.set_ylabel("Score")
    
#     canvas = FigureCanvasTkAgg(fig, master=right_frame)
#     canvas_widget = canvas.get_tk_widget()
#     canvas_widget.grid(row=0, column=0)


def run_model():

    datasets=sorted(["CM1", "MW1", "PC1", "PC3", "PC4"])
    dataset_name = dataset_combobox.get()
    Xtarget, Ltarget = ut.read_dataset("data/PROMISE/", dataset_name=dataset_name) # data/AEEEM/ , data/PROMISE/ ,or data/SOFTLAB/
    Ltarget[Ltarget > 1] = 1
    Xtarget=Xtarget

    
    Xsource=np.zeros((1,Xtarget.shape[1]))
    Lsource=[0]
    source_w=[0]
    sn=0
    Xt_mean = Xtarget.mean(axis=0)

    dists=[]
    Xs_mean=0
    for dss in datasets:
        if dss!=dataset_name:
            sn+=1
            X, L = ut.read_dataset("data/PROMISE/", dataset_name=dss)
            # snv=np.ones((L.shape[0]))*sn
            Xs_mean += X.mean(axis=0)

            Xsource= np.concatenate((Xsource, X), axis=0)

            Lsource= np.concatenate((Lsource, L), axis=0)

    Xsource=Xsource[1:,:]
    Lsource=Lsource[1:]
    source_w=source_w[1:]
    Lsource[Lsource > 1] = 1
    print(dataset_name + ' Start!')
    Run(Xsource, Lsource,source_w, Xtarget, Ltarget)

def exit_app():
    root.quit()
    root.destroy()

# root = tk.Tk()
# root.geometry("1000x700")
# root.title("A NOVEL CROSS-PROJECT SOFTWARE DEFECT PREDICTION ALGORITHM BASED ON TRANSFER LEARNING")

# header_frame = ttk.Frame(root)
# header_frame.pack(pady=10)

# # Add text saying "Project" in the header part with center alignment
# header_label = tk.Label(header_frame, text="Project", font=("Arial", 16))
# header_label.pack()


# left_frame = ttk.Frame(root)
# left_frame.grid(row=0, column=0, padx=10, pady= 10)

# right_frame = ttk.Frame(root)
# right_frame.grid(row=0, column=1, padx=10, pady=10)

# dataset_combobox = ttk.Combobox(left_frame, values=["CM1", "MW1", "PC1", "PC3", "PC4"], state="readonly")
# dataset_combobox.set("CM1")  # Set the default dataset name
# dataset_combobox.grid(row=0, column=0)

# select_button = ttk.Button(left_frame, text="Select Dataset", command=select_dataset)
# select_button.grid(row=0, column=1)

# run_button = ttk.Button(left_frame, text="Run Model", command=run_model)
# run_button.grid(row=1, column=0)

# exit_button = ttk.Button(left_frame, text="Exit", command=exit_app)
# exit_button.grid(row=2, column=0)

# result_label = ttk.Label(left_frame, text="")
# result_label.grid(row=3, column=0)

# datasets = sorted(["CM1", "MW1", "PC1", "PC3", "PC4"])
# ds = datasets[0]  # Initialize with the first dataset

# warnings.filterwarnings('ignore')
# def on_closing():
#     root.destroy()

# root.protocol("WM_DELETE_WINDOW", on_closing)

# root.config()
# root.mainloop()


root = tk.Tk()
root.geometry("1150x800")
root.title("A NOVEL CROSS-PROJECT SOFTWARE DEFECT PREDICTION ALGORITHM BASED ON TRANSFER LEARNING")

# Header Part
header_frame = ttk.Frame(root)
header_frame.pack(pady=10)

# Add text saying "Project" in the header part with center alignment
header_label = ttk.Label(header_frame, text="A NOVEL CROSS-PROJECT SOFTWARE DEFECT PREDICTION ALGORITHM BASED ON TRANSFER LEARNING", font=("Arial", 14))
header_label.pack()

# Content Area
content_frame = ttk.Frame(root)
content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Left Frame
left_frame = ttk.Frame(content_frame)
left_frame.pack(side="left", padx=10, pady=10)

# Right Frame
right_frame = ttk.Frame(content_frame)
right_frame.pack(side="right", padx=10, pady=10)

# Dataset Selection Section
dataset_label = ttk.Label(left_frame, text="Select Dataset:")
dataset_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

dataset_combobox = ttk.Combobox(left_frame, values=["CM1", "MW1", "PC1", "PC3", "PC4"], state="readonly")
dataset_combobox.set("CM1")  # Set the default dataset name
dataset_combobox.grid(row=0, column=1, padx=5, pady=5)

# Model Running Section
run_button = ttk.Button(left_frame, text="Run Model", command=run_model)
run_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

# Exit Button
exit_button = ttk.Button(left_frame, text="Exit", command=exit_app)
exit_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

# Result Labels
result_label = ttk.Label(left_frame, text="Results:",font=("Serif",16))
result_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")


# Result Labels
algo_label = ttk.Label(left_frame, text="TransferBoost",font=("Serif",14))
algo_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

auc_label = ttk.Label(left_frame, text="AUC Score: -  ",font=("Serif",14))
auc_label.grid(row=6, column=0, columnspan=2, padx=5, pady=2, sticky="ew")

f1_label = ttk.Label(left_frame, text="F1 Score: -  ",font=("Serif",14))
f1_label.grid(row=7, column=0, columnspan=2, padx=5, pady=2, sticky="ew")

accuracy_label = ttk.Label(left_frame, text="Accuracy: -  ",font=("Serif",14))
accuracy_label.grid(row=8, column=0, columnspan=2, padx=5, pady=2, sticky="ew")

# Set up closing protocol
def on_closing():
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()