import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

version = 'DQN_Mel'

len = pd.read_csv(f"PlotData\{version}_Len.csv")
rew = pd.read_csv(f"PlotData\{version}_Rew.csv")

info_len = len[['Step', 'Value']]
info_rew = rew[['Step', 'Value']]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))


sns.scatterplot(x = "Step", y = "Value",data=info_len, alpha= 0.4, ax=axes[0])
sns.regplot(x = "Step", y = "Value",data=info_len, order=4, color='r', 
            robust=False, scatter=False, ax=axes[0]).set(title='Episodic Lenght', 
                                                         xlabel= 'Episode', ylabel= 'Game Lenght')
plt.legend(labels=["Actual Lenght","Trend"])

sns.scatterplot(x = "Step", y = "Value",data=info_rew, alpha= 0.4, ax=axes[1])
sns.regplot(x = "Step", y = "Value",data=info_rew, order=4, 
            robust=False, scatter=False, ax=axes[1], color='r').set(title='Episodic Reward', 
                                                         xlabel= 'Episode', ylabel= 'Game Reward')
plt.legend(labels=["Actual Reward","Trend"])
sns.set(style='dark',)

plt.show()

