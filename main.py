import os
import subprocess

def run_pca_analysis():
    print("Running PCA Analysis...")
    os.chdir("./pca_analysis")
    subprocess.run(["python", "pca_analysis.py"])
    os.chdir("..")

def run_logistic_regression():
    print("Running Logistic Regression tasks...")
    os.chdir("./logistic_regression")
    for script in ["q2_1.py", "q2_2.py", "q2_3_4.py", "q2_5.py"]:
        subprocess.run(["python", script])
    os.chdir("..")

if __name__ == "__main__":
    print("Running Assignment Tasks...")
    run_pca_analysis()
    run_logistic_regression()
