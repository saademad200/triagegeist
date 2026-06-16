# 2. Data Analysis and Visualization


## 2.1 Data Analysis


```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/kaggle/input/competitions/triagegeist/patient_history.csv')
print("✅ Data loaded successfully!")
print("Shape:", df.shape)
```


```python
# ─── BASE CLASS ──────────────────────────────────────────
# Create base classes needed to analyze the data of all datasets
# Can be the base class then sub classes including
# class for overview of data, class for plotting visual (line, bar, pie,...)
# class for normalization...

class DataAnalysisBase(ABC):
    
    def __init__(self, dataframe):
        self.df = dataframe
    
    @abstractmethod
    def analyze(self):
        pass
    
    @abstractmethod
    def visualize(self):
        pass


# ─── SUB CLASS 1: Data Overview ──────────────────────────
class DataOverview(DataAnalysisBase):
    
    def analyze(self):
        print("=" * 60)
        print("📋 DATA OVERVIEW")
        print("=" * 60)
        print(f"\nShape: {self.df.shape}")
        print(f"Total patients: {len(self.df)}")
        print(f"\nColumn Names:")
        for col in self.df.columns.tolist():
            print(f"   → {col}")
        print(f"\nMissing Values:")
        total = self.df.isnull().sum().sum()
        if total == 0:
            print("   ✅ No missing values found!")
        else:
            print(self.df.isnull().sum())
    
    def visualize(self):
        print("\nFirst 5 rows of dataset:")
        print(self.df.head())

print("✅ Classes defined successfully!")
```


```python
overview = DataOverview(df)
overview.analyze()
overview.visualize()
    
```

From the data overview, the dataset contains 100,000 patients
and 26 columns representing different medical conditions.
All values are binary (0 or 1), meaning each condition is
either present or absent. No missing values were found,
confirming the dataset is clean and ready for analysis.

## 2.2 Data Visualization



```python
# ─── SUB CLASS 2: Data Visualization ─────────────────────
# For the plotting of different graphs types such as line, bar, pie, etc.

class DataVisualization(DataAnalysisBase):

    def analyze(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        condition_counts = self.df[condition_columns].sum()
        print("=" * 60)
        print("📊 CONDITION ANALYSIS")
        print("=" * 60)
        print(f"\nMost common condition:  {condition_counts.idxmax()}")
        print(f"Least common condition: {condition_counts.idxmin()}")
        print(f"\nTop 5 conditions:")
        print(condition_counts.sort_values(ascending=False).head())

    def visualize(self):
        self.plot_bar()
        self.plot_bar_percentage()
        self.plot_heatmap()
        self.plot_line()
        self.plot_histogram()

    def plot_bar(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        condition_counts = self.df[condition_columns].sum().sort_values(ascending=False)
        plt.figure(figsize=(14, 6))
        ax = sns.barplot(x=condition_counts.index, 
                         y=condition_counts.values, palette='Reds_r')
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha='center', va='bottom', fontsize=8)
        plt.title('Number of Patients with Each Medical Condition')
        plt.xlabel('Medical Condition')
        plt.ylabel('Number of Patients')
        plt.xticks(rotation=90)
        plt.show()

    def plot_bar_percentage(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        condition_pct = (self.df[condition_columns].sum() / 
                        len(self.df)).sort_values(ascending=False)
        plt.figure(figsize=(14, 6))
        ax = sns.barplot(x=condition_pct.index, 
                         y=condition_pct.values, palette='Blues_r')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                       (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha='center', va='bottom', fontsize=8)
        plt.title('Percentage of Patients with Each Medical Condition')
        plt.xlabel('Medical Condition')
        plt.ylabel('Proportion of Patients')
        plt.xticks(rotation=90)
        plt.show()

    def plot_heatmap(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        plt.figure(figsize=(16, 12))
        sns.heatmap(self.df[condition_columns].corr(), 
                    annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Between Medical Conditions')
        plt.show()

    def plot_line(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        condition_counts = self.df[condition_columns].sum()
        plt.figure(figsize=(14, 6))
        plt.plot(condition_counts.index, 
                 condition_counts.values, 
                 marker='o', color='steelblue')
        plt.title('Patient Count Per Condition')
        plt.xlabel('Medical Condition')
        plt.ylabel('Number of Patients')
        plt.xticks(rotation=90)
        plt.show()

    def plot_histogram(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        self.df[condition_columns].hist(bins=50, figsize=(12, 8))
        plt.suptitle('Distribution of All Medical Conditions')
        plt.show()

print("✅ DataVisualization class defined!")
```


```python
viz = DataVisualization(df)
viz.analyze()
```

From the condition analysis, hx_hypertension is the most
common condition while hx_hiv is the least common.
This gives us a clear picture of which conditions dominate
the emergency patient population.


```python
viz.plot_bar()
```

From the bar chart, hx_hypertension has the highest patient
count making it the most common condition in this dataset.
This suggests that cardiovascular conditions are the most
prevalent among emergency room patients.


```python
viz.plot_bar_percentage()

```

From the percentage chart, the difference between conditions
is consistent from one condition to another, suggesting a
balanced construction of the dataset. Majority of patients
carry at least one chronic condition when arriving at the
emergency room.


```python
viz.plot_heatmap()
```

From the heatmap, most medical conditions show very low
correlation with each other (close to 0.00), meaning they
appear independently. However a few condition pairs show
slightly higher correlation, suggesting certain diseases
tend to co-exist in the same patients.


```python
viz.plot_line()
```

From the line chart, patient count across conditions is
uneven with some conditions spiking significantly higher.
This confirms that certain chronic conditions are far more
common in emergency patients and should be given higher
priority weight in the triage prediction model.


```python
viz.plot_histogram()

```

From the histogram, all conditions show a clear binary
distribution concentrated at 0 and 1. This confirms the
dataset is correctly formatted as binary medical history
data with no unusual distributions detected.

## 2.3 Result Interpretation


```python
# ─── SUB CLASS 3: Result Interpretation ──────────────────
# Interpret and communicate findings to stakeholders

class ResultInterpretation(DataAnalysisBase):

    def analyze(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        condition_counts = self.df[condition_columns].sum()
        total_patients = len(self.df)
        most_common = condition_counts.idxmax()
        least_common = condition_counts.idxmin()
        most_common_pct = round(condition_counts.max() / total_patients * 100, 2)
        least_common_pct = round(condition_counts.min() / total_patients * 100, 2)
        self.df['total_conditions'] = self.df[condition_columns].sum(axis=1)
        avg_conditions = round(self.df['total_conditions'].mean(), 2)

        print("=" * 60)
        print("📋 RESULT INTERPRETATION REPORT")
        print("=" * 60)
        print("\n🔍 1. CONTEXT OF ORIGINAL PROBLEM")
        print(f"   Total patients analyzed: {total_patients}")
        print(f"   Total conditions tracked: {len(condition_columns)}")
        print(f"   Average conditions per patient: {avg_conditions}")
        print("\n💡 2. KEY FINDINGS")
        print(f"   Most common:  {most_common} → {most_common_pct}% of patients")
        print(f"   Least common: {least_common} → {least_common_pct}% of patients")
        print("\n✅ 3. ACTIONABLE RECOMMENDATIONS")
        print(f"   → Prioritize screening for {most_common}")
        print(f"   → Flag patients with many conditions as high risk")
        print(f"   → ML model should weight {most_common} heavily")
        print("\n📢 4. COMMUNICATING TO STAKEHOLDERS")
        print(f"   → {most_common_pct}% of ER patients have {most_common}")
        print(f"   → Average patient carries {avg_conditions} conditions")
        print(f"   → This supports AI triage prioritization")
        print("\n🔄 5. MONITORING & IMPROVEMENT")
        print("   → Re-run analysis weekly with new data")
        print("   → Compare condition trends month over month")
        print("   → Update ML model if new patterns emerge")
        print("=" * 60)

    def visualize(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        self.df['total_conditions'] = self.df[condition_columns].sum(axis=1)
        plt.figure(figsize=(10, 5))
        sns.countplot(x='total_conditions', data=self.df, palette='Purples_d')
        plt.title('How Many Conditions Each Patient Has')
        plt.xlabel('Number of Conditions')
        plt.ylabel('Number of Patients')
        plt.show()

    def save_report(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        try:
            self.df[condition_columns].sum().to_csv(
                'condition_report.csv', header=['patient_count'])
            self.df.describe().to_csv('patient_history_summary.csv')
            print("✅ Reports saved successfully!")
        except Exception as e:
            print(f"Error saving: {e}")

print("✅ ResultInterpretation class defined!")
```


```python
interpret = ResultInterpretation(df)
interpret.analyze()
```

From the interpretation report, hx_hypertension dominates
with the highest percentage of patients affected. The average
patient carries multiple conditions simultaneously, confirming
that medical history is a strong indicator for triage acuity.


```python
interpret.visualize()
```

From the conditions count chart, majority of patients carry
between 2 to 4 conditions simultaneously. Patients with more
than 5 conditions should be flagged as high risk in the triage
system as they are more likely to require urgent care.


```python
interpret.save_report()
```

The analysis report has been saved successfully. These findings
will be used by the team to improve the Lifier AI triage model
by providing evidence-based weights for medical history features.
Continuous monitoring is recommended as new patient data arrives.
