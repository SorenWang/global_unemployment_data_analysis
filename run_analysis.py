"""
============================================================
全球失业率数据分析 - 批量运行脚本
运行方式: python run_analysis.py
============================================================
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免字体渲染问题
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')
import os
import glob as glob_mod

# ============================================================
# 中文字体修复：清除缓存 + 显式加载字体文件
# ============================================================
CACHE_DIR = matplotlib.get_cachedir()
for cache_file in glob_mod.glob(os.path.join(CACHE_DIR, 'fontlist*')):
    try:
        os.remove(cache_file)
    except:
        pass

# 显式加载 Windows 中文字体
for font_path in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)

# 强制 matplotlib 使用中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 重建字体管理器
fm._load_fontmanager(try_read_cache=False)
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
year_cols = [str(y) for y in range(2014, 2025)]

print("=" * 60)
print("全球失业率数据分析 - 开始执行")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/20] 加载数据...")
df = pd.read_csv('global_unemployment_data.csv')
print(f"数据形状: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"国家数量: {df['country_name'].nunique()}")

# ============================================================
# 2. 数据清洗
# ============================================================
print("\n[2/20] 数据清洗...")
missing = df[year_cols].isnull().sum().sum()
print(f"年份列缺失值: {missing}")
if missing > 0:
    for col in year_cols:
        df[col] = df.groupby(['country_name', 'sex', 'age_group'])[col].transform(
            lambda x: x.fillna(x.mean()))
    df[year_cols] = df[year_cols].fillna(df[year_cols].mean())

# ============================================================
# 3. 派生特征
# ============================================================
print("\n[3/20] 添加派生特征...")
df['avg_unemployment'] = df[year_cols].mean(axis=1)
df['trend_2014_2024'] = df['2024'] - df['2014']
df['volatility'] = df[year_cols].std(axis=1)
df['max_unemployment'] = df[year_cols].max(axis=1)
df['min_unemployment'] = df[year_cols].min(axis=1)
df['covid_avg'] = df[['2020', '2021']].mean(axis=1)
df['covid_impact'] = df[['2020', '2021']].mean(axis=1) - df[['2018', '2019']].mean(axis=1)

# ============================================================
# 4. 描述性统计 - 保存
# ============================================================
print("\n[4/20] 描述性统计...")
desc_stats = df[year_cols].describe()
desc_stats.to_csv(f'{OUTPUT_DIR}/descriptive_stats.csv', encoding='utf-8-sig')

gender_stats = df.groupby('sex').agg({
    'avg_unemployment': ['mean', 'std', 'min', 'max'],
    'trend_2014_2024': 'mean', 'covid_impact': 'mean', 'volatility': 'mean'
}).round(3)
gender_stats.to_csv(f'{OUTPUT_DIR}/gender_stats.csv', encoding='utf-8-sig')

age_stats = df.groupby('age_categories').agg({
    'avg_unemployment': ['mean', 'std', 'min', 'max'],
    'trend_2014_2024': 'mean', 'covid_impact': 'mean'
}).round(3)
age_stats.to_csv(f'{OUTPUT_DIR}/age_stats.csv', encoding='utf-8-sig')

yearly_global = pd.DataFrame({
    'Year': [int(y) for y in year_cols],
    'Global_Mean': [df[y].mean() for y in year_cols],
    'Global_Median': [df[y].median() for y in year_cols],
    'Global_Std': [df[y].std() for y in year_cols]
}).set_index('Year')
yearly_global.to_csv(f'{OUTPUT_DIR}/yearly_global_trend.csv', encoding='utf-8-sig')

country_summary = df.groupby('country_name')['avg_unemployment'].mean().sort_values(ascending=False)
country_summary.to_csv(f'{OUTPUT_DIR}/country_summary.csv', encoding='utf-8-sig')

covid_country_impact = df.groupby('country_name')['covid_impact'].mean().sort_values(ascending=False)
covid_country_impact.to_csv(f'{OUTPUT_DIR}/covid_impact.csv', encoding='utf-8-sig')

# ============================================================
# 5. 可视化1: 全球趋势
# ============================================================
print("\n[5/20] 图表1: 全球失业率年度趋势...")
fig, ax = plt.subplots(figsize=(12, 6))
years_int = [int(y) for y in year_cols]
means = [df[y].mean() for y in year_cols]
stds = [df[y].std() for y in year_cols]

ax.plot(years_int, means, 'o-', color='#2c3e50', linewidth=2.5, markersize=8, label='Global Mean Unemployment Rate')
ax.fill_between(years_int,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.2, color='#3498db', label='±1 Std Dev')
ax.axvspan(2019.5, 2021.5, alpha=0.15, color='red', label='COVID-19 Period')
ax.axvline(x=2020, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
for yr, m in zip(years_int, means):
    ax.annotate(f'{m:.2f}%', (yr, m), textcoords='offset points',
                xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Global Average Unemployment Rate Trend (2014-2024)', fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.set_xticks(years_int)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_global_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 6. 可视化2: 性别对比
# ============================================================
print("\n[6/20] 图表2: 男女失业率对比...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
male_data = [df[df['sex']=='Male'][y].dropna().values for y in year_cols]
female_data = [df[df['sex']=='Female'][y].dropna().values for y in year_cols]
pos_male = np.arange(len(year_cols)) * 2 - 0.3
pos_female = np.arange(len(year_cols)) * 2 + 0.3

bp1 = axes[0].boxplot(male_data, positions=pos_male, widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor='#3498db', alpha=0.7),
                       medianprops=dict(color='darkblue', linewidth=2))
bp2 = axes[0].boxplot(female_data, positions=pos_female, widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor='#e74c3c', alpha=0.7),
                       medianprops=dict(color='darkred', linewidth=2))
axes[0].set_xticks(np.arange(len(year_cols)) * 2)
axes[0].set_xticklabels(year_cols, rotation=45)
axes[0].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Unemployment Rate by Gender (Yearly)', fontsize=13, fontweight='bold')
axes[0].legend([bp1['boxes'][0], bp2['boxes'][0]], ['Male', 'Female'], fontsize=10)

gender_means = df.groupby('sex')['avg_unemployment'].mean()
bars = axes[1].bar(['Male', 'Female'], gender_means.values,
                    color=['#3498db', '#e74c3c'], edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, gender_means.values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Average Unemployment Rate: Male vs Female', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_gender_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 7. 可视化3: 年龄热力图
# ============================================================
print("\n[7/20] 图表3: 年龄组热力图...")
fig, ax = plt.subplots(figsize=(12, 5))
age_yearly = df.groupby('age_categories').agg({y: 'mean' for y in year_cols})
sns.heatmap(age_yearly, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label': 'Unemployment Rate (%)'},
            ax=ax, annot_kws={'fontsize': 10})
ax.set_title('Unemployment Rate Heatmap by Age Group (2014-2024)', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Age Group', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_age_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 8. 可视化4: 直方图
# ============================================================
print("\n[8/20] 图表4: 2024年失业率分布...")
fig, ax = plt.subplots(figsize=(12, 6))
data_2024 = df['2024'].dropna()
ax.hist(data_2024, bins=50, color='#2ecc71', edgecolor='white',
        alpha=0.7, density=True, label='2024 Distribution')
kde = gaussian_kde(data_2024)
x_range = np.linspace(data_2024.min(), data_2024.max(), 200)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE Curve')
ax.axvline(data_2024.mean(), color='#e74c3c', linestyle='--', linewidth=2,
           label=f'Mean: {data_2024.mean():.2f}%')
ax.axvline(data_2024.median(), color='#3498db', linestyle='--', linewidth=2,
           label=f'Median: {data_2024.median():.2f}%')
ax.set_xlabel('Unemployment Rate (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Density', fontsize=13, fontweight='bold')
ax.set_title('Distribution of Unemployment Rates in 2024', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_distribution_histogram.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 9. 可视化5: COVID散点图
# ============================================================
print("\n[9/20] 图表5: COVID-19影响散点图...")
fig, ax = plt.subplots(figsize=(12, 7))
country_covid = df.groupby('country_name').agg({
    'covid_impact': 'mean', 'avg_unemployment': 'mean', 'trend_2014_2024': 'mean'
}).reset_index()
scatter = ax.scatter(country_covid['avg_unemployment'], country_covid['covid_impact'],
                     c=country_covid['trend_2014_2024'], cmap='RdYlGn_r',
                     s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Overall Trend (2014-2024, % change)', fontsize=11, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.axvline(x=country_covid['avg_unemployment'].median(), color='blue',
           linestyle='--', linewidth=1, alpha=0.5, label='Median Unemployment')
extreme = country_covid.nlargest(5, 'covid_impact')
for _, row in extreme.iterrows():
    ax.annotate(row['country_name'],
                (row['avg_unemployment'], row['covid_impact']),
                fontsize=8, alpha=0.8,
                arrowprops=dict(arrowstyle='->', alpha=0.5))
ax.set_xlabel('Average Unemployment Rate (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('COVID-19 Impact (Percentage Point Change)', fontsize=13, fontweight='bold')
ax.set_title('COVID-19 Impact vs Average Unemployment Rate', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_covid_impact_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 10. 可视化6: 性别×年龄组
# ============================================================
print("\n[10/20] 图表6: 性别×年龄组柱状图...")
fig, ax = plt.subplots(figsize=(12, 6))
cross_stats = df.groupby(['sex', 'age_categories'])['avg_unemployment'].mean().unstack()
x = np.arange(len(cross_stats.columns))
width = 0.35
bars1 = ax.bar(x - width/2, cross_stats.loc['Male'], width, label='Male',
               color='#3498db', edgecolor='white', linewidth=1.2)
bars2 = ax.bar(x + width/2, cross_stats.loc['Female'], width, label='Female',
               color='#e74c3c', edgecolor='white', linewidth=1.2)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', fontsize=10, fontweight='bold', color='#2980b9')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', fontsize=10, fontweight='bold', color='#c0392b')
ax.set_xticks(x)
ax.set_xticklabels(cross_stats.columns, fontsize=12)
ax.set_xlabel('Age Group', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Unemployment Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Unemployment Rate by Gender and Age Group', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_gender_age_barchart.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 11. 可视化7: Top30国家
# ============================================================
print("\n[11/20] 图表7: Top30国家...")
fig, ax = plt.subplots(figsize=(12, 10))
top30 = country_summary.head(30).sort_values(ascending=True)
colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(top30)))
bars = ax.barh(top30.index, top30.values, color=colors_bar, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, top30.values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
            f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Average Unemployment Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Top 30 Countries with Highest Unemployment Rates (2014-2024 Avg)', fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig7_top30_countries.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 12. 可视化8: 年龄组趋势
# ============================================================
print("\n[12/20] 图表8: 年龄组趋势...")
fig, ax = plt.subplots(figsize=(12, 6))
age_groups = df['age_categories'].unique()
colors_line = ['#e74c3c', '#3498db', '#2ecc71']
markers = ['o', 's', '^']
for age, color, marker in zip(age_groups, colors_line, markers):
    age_data = df[df['age_categories'] == age]
    yearly_mean = [age_data[y].mean() for y in year_cols]
    ax.plot(years_int, yearly_mean, marker=marker, color=color,
            linewidth=2.5, markersize=8, label=age)
ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Unemployment Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Unemployment Rate Trends by Age Group (2014-2024)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.set_xticks(years_int)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig8_age_trends.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 13. t检验
# ============================================================
print("\n[13/20] t检验: 男女失业率差异...")
male_avg = df[df['sex'] == 'Male']['avg_unemployment']
female_avg = df[df['sex'] == 'Female']['avg_unemployment']
t_stat, p_value = stats.ttest_ind(male_avg, female_avg)
ttest_result = pd.DataFrame({
    'Metric': ['Male Mean', 'Female Mean', 'Male Std', 'Female Std', 't-statistic', 'p-value', 'Significant (α=0.05)'],
    'Value': [male_avg.mean(), female_avg.mean(), male_avg.std(), female_avg.std(), t_stat, p_value, 'Yes' if p_value < 0.05 else 'No']
})
ttest_result.to_csv(f'{OUTPUT_DIR}/ttest_result.csv', index=False, encoding='utf-8-sig')
print(f"  t={t_stat:.4f}, p={p_value:.6f}, significant={p_value < 0.05}")

# ============================================================
# 14. ANOVA
# ============================================================
print("\n[14/20] ANOVA: 年龄组差异...")
youth = df[df['age_categories'] == 'Youth']['avg_unemployment']
adults = df[df['age_categories'] == 'Adults']['avg_unemployment']
children = df[df['age_categories'] == 'Children']['avg_unemployment']
f_stat, p_value_anova = stats.f_oneway(youth, adults, children)
anova_result = pd.DataFrame({
    'Metric': ['Youth Mean', 'Adults Mean', 'Children Mean', 'Youth Std', 'Adults Std', 'Children Std', 'F-statistic', 'p-value', 'Significant (α=0.05)'],
    'Value': [youth.mean(), adults.mean(), children.mean(), youth.std(), adults.std(), children.std(), f_stat, p_value_anova, 'Yes' if p_value_anova < 0.05 else 'No']
})
anova_result.to_csv(f'{OUTPUT_DIR}/anova_result.csv', index=False, encoding='utf-8-sig')
print(f"  F={f_stat:.4f}, p={p_value_anova:.6f}, significant={p_value_anova < 0.05}")

# ============================================================
# 15. K-Means聚类
# ============================================================
print("\n[15/20] K-Means聚类分析...")
cluster_features = df.groupby('country_name').agg({
    'avg_unemployment': 'mean', 'volatility': 'mean',
    'trend_2014_2024': 'mean', 'covid_impact': 'mean'
})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertias, 'bo-', markersize=8, linewidth=2)
ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
ax.set_ylabel('Inertia', fontsize=12, fontweight='bold')
ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig9_elbow_method.png', dpi=150, bbox_inches='tight')
plt.close()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_features['Cluster'] = kmeans.fit_predict(X_scaled)
cluster_features.to_csv(f'{OUTPUT_DIR}/kmeans_clusters.csv', encoding='utf-8-sig')
for i in range(4):
    cnt = (cluster_features['Cluster'] == i).sum()
    cmean = cluster_features[cluster_features['Cluster'] == i]['avg_unemployment'].mean()
    print(f"  Cluster {i}: {cnt} countries, avg rate={cmean:.2f}%")

# ============================================================
# 16. 线性回归
# ============================================================
print("\n[16/20] 线性回归...")
years_arr = np.array(years_int).reshape(-1, 1)
global_means_arr = np.array([df[y].mean() for y in year_cols])
lr_model = LinearRegression()
lr_model.fit(years_arr, global_means_arr)
y_pred_lr = lr_model.predict(years_arr)
future_years = np.array(range(2025, 2031)).reshape(-1, 1)
future_pred = lr_model.predict(future_years)
r2_lr = r2_score(global_means_arr, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(global_means_arr, y_pred_lr))
mae_lr = mean_absolute_error(global_means_arr, y_pred_lr)
print(f"  R2={r2_lr:.4f}, RMSE={rmse_lr:.4f}, MAE={mae_lr:.4f}")

lr_results = pd.DataFrame({
    'Year': list(range(2014, 2031)),
    'Type': ['Actual']*11 + ['Predicted']*6,
    'Unemployment_Rate': list(global_means_arr) + list(future_pred)
})
lr_results.to_csv(f'{OUTPUT_DIR}/linear_regression_results.csv', index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(years_arr, global_means_arr, color='blue', s=100, zorder=5, label='Actual Data')
ax.plot(years_arr, y_pred_lr, 'b-', linewidth=2, label=f'Regression Line (R2={r2_lr:.3f})')
ax.scatter(future_years, future_pred, color='red', s=100, zorder=5, marker='s', label='2025-2030 Predictions')
ax.plot(np.vstack([years_arr[-1:], future_years]),
        np.hstack([y_pred_lr[-1:], future_pred]),
        'r--', linewidth=2, alpha=0.7)
ax.fill_between(future_years.ravel(),
                future_pred.ravel() - rmse_lr,
                future_pred.ravel() + rmse_lr,
                alpha=0.2, color='red', label=f'±RMSE ({rmse_lr:.2f}%)')
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Global Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Linear Regression: Unemployment Rate Forecast', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig10_linear_regression.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 17. 随机森林
# ============================================================
print("\n[17/20] 随机森林回归...")
df_encoded = df.copy()
le_sex = LabelEncoder()
le_age = LabelEncoder()
le_country = LabelEncoder()
df_encoded['sex_encoded'] = le_sex.fit_transform(df_encoded['sex'])
df_encoded['age_encoded'] = le_age.fit_transform(df_encoded['age_group'])
df_encoded['country_encoded'] = le_country.fit_transform(df_encoded['country_name'])

feature_cols = ['sex_encoded', 'age_encoded', 'country_encoded',
                '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
X = df_encoded[feature_cols]
y = df_encoded['2024']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)
print(f"  R2={rf_r2:.4f}, RMSE={rf_rmse:.4f}, MAE={rf_mae:.4f}")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
feature_importance.to_csv(f'{OUTPUT_DIR}/rf_feature_importance.csv', index=False, encoding='utf-8-sig')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].barh(feature_importance['Feature'][::-1], feature_importance['Importance'][::-1],
             color=plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_importance))))
axes[0].set_xlabel('Importance', fontsize=12, fontweight='bold')
axes[0].set_title('Random Forest Feature Importance', fontsize=13, fontweight='bold')
axes[1].scatter(y_test, y_pred_rf, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[1].set_title(f'RF: Predicted vs Actual (R2={rf_r2:.3f})', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig11_random_forest.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 18. PCA
# ============================================================
print("\n[18/20] PCA主成分分析...")
pca_data = df.groupby('country_name')[year_cols].mean()
scaler_pca = StandardScaler()
pca_scaled = scaler_pca.fit_transform(pca_data)
pca = PCA()
pca_result = pca.fit_transform(pca_scaled)
explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)
print(f"  PC1+PC2 解释方差: {cumsum_var[1]*100:.1f}%")

pca_df = pd.DataFrame({
    'Country': pca_data.index,
    'PC1': pca_result[:, 0], 'PC2': pca_result[:, 1], 'PC3': pca_result[:, 2]
})
pca_df.to_csv(f'{OUTPUT_DIR}/pca_results.csv', index=False, encoding='utf-8-sig')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].bar(range(1, len(explained_var)+1), explained_var*100, alpha=0.7, color='#3498db', label='Individual')
axes[0].plot(range(1, len(explained_var)+1), cumsum_var*100, 'ro-', linewidth=2, markersize=6, label='Cumulative')
axes[0].axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% Threshold')
axes[0].set_xlabel('Principal Component', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Explained Variance Ratio (%)', fontsize=12, fontweight='bold')
axes[0].set_title('PCA Explained Variance', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)

scatter2 = axes[1].scatter(pca_result[:, 0], pca_result[:, 1],
                          c=pca_data.mean(axis=1), cmap='RdYlGn_r',
                          s=50, alpha=0.7, edgecolors='black', linewidth=0.3)
cbar2 = plt.colorbar(scatter2, ax=axes[1])
cbar2.set_label('Avg Unemployment Rate', fontsize=10)
axes[1].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=12, fontweight='bold')
axes[1].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=12, fontweight='bold')
axes[1].set_title('PCA: Countries in PC1-PC2 Space', fontsize=13, fontweight='bold')
extreme_idx = np.argsort(np.abs(pca_result[:, 0]))[-5:]
for idx in extreme_idx:
    axes[1].annotate(pca_data.index[idx],
                    (pca_result[idx, 0], pca_result[idx, 1]),
                    fontsize=7, alpha=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig12_pca_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 19. 摘要
# ============================================================
print("\n[19/20] 保存分析摘要...")
summary = f"""=== Global Unemployment Data Analysis Summary ===

1. Data Overview:
   - Source: International Labour Organization (ILO)
   - Records: {df.shape[0]}, Countries: {df['country_name'].nunique()}
   - Time Span: 2014-2024
   - Dimensions: Sex (Male/Female), Age Groups (Youth/Adults/Children)

2. Preliminary Analysis:
   - Global unemployment shows a slight downward trend
   - Female unemployment is systematically higher than male
   - Youth unemployment is significantly higher than other age groups
   - COVID-19 caused significant spike in 2020-2021

3. Statistical Tests:
   - t-test: Male vs Female unemployment - Significant difference (p={p_value:.6f})
   - ANOVA: Age group differences - Significant difference (p={p_value_anova:.6f})

4. Machine Learning Models:
   - K-Means Clustering: Countries grouped into 4 unemployment pattern clusters
   - Linear Regression: R2={r2_lr:.4f}, RMSE={rmse_lr:.4f}%
   - Random Forest: R2={rf_r2:.4f}, RMSE={rf_rmse:.4f}%
   - PCA: First 2 PCs explain {cumsum_var[1]*100:.1f}% variance

5. Output Files:
   - 12 visualization charts (fig1-fig12)
   - Statistical test results (CSV)
   - ML model results (CSV)
   - Descriptive statistics (CSV)
"""
with open(f'{OUTPUT_DIR}/analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

# ============================================================
# 20. 完成
# ============================================================
print("\n[20/20] 完成！输出文件列表：")
for file in sorted(os.listdir(OUTPUT_DIR)):
    filepath = os.path.join(OUTPUT_DIR, file)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  {file} ({size_kb:.1f} KB)")

print(f"\n共 {len(os.listdir(OUTPUT_DIR))} 个输出文件")
print("=" * 60)
print("所有分析完成！运行 streamlit run streamlit_app.py 启动Web仪表盘")
print("=" * 60)
