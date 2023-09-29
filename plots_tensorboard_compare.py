import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

version1 = 'DQN_Mel'
version2 = 'DQN_LR0.001'

len1 = pd.read_csv(f"PlotData\{version1}_Len.csv")
rew1 = pd.read_csv(f"PlotData\{version1}_Rew.csv")

len2 = pd.read_csv(f"PlotData\{version2}_Len.csv")
rew2 = pd.read_csv(f"PlotData\{version2}_Rew.csv")

info_len1 = len1[['Step', 'Value']]
info_rew1 = rew1[['Step', 'Value']]

info_len2 = len2[['Step', 'Value']]
info_rew2 = rew2[['Step', 'Value']]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))


sns.regplot(x = "Step", y = "Value",data=info_len1, order=4, color='r', 
            robust=False, scatter=False, ax=axes[0])
sns.regplot(x = "Step", y = "Value",data=info_len2, order=4, color='r', 
            robust=False, scatter=False, ax=axes[0]).set(title='Episodic Lenght', 
                                                         xlabel= 'Episode', ylabel= 'Game Lenght')
plt.legend(labels=["Actual Lenght","Trend"])

sns.scatterplot(x = "Step", y = "Value",data=info_rew1, alpha= 0.4, ax=axes[1])
sns.regplot(x = "Step", y = "Value",data=info_rew1, order=4, 
            robust=False, scatter=False, ax=axes[1], color='r').set(title='Episodic Reward', 
                                                         xlabel= 'Episode', ylabel= 'Game Reward')
plt.legend(labels=["Actual Reward","Trend"])
sns.set(style='dark',)

plt.show()

