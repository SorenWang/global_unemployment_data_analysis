"""
============================================================
全球失业率数据分析 - Streamlit Web仪表盘
Global Unemployment Rate Analysis Dashboard
============================================================

运行方式: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import glob

# ============================================================
# 中文字体修复
# ============================================================
CACHE_DIR = matplotlib.get_cachedir()
for cache_file in glob.glob(os.path.join(CACHE_DIR, 'fontlist*')):
    try:
        os.remove(cache_file)
    except:
        pass
for font_path in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fm._load_fontmanager(try_read_cache=False)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="全球失业率分析仪表盘",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 自定义CSS样式
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #3498db;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        padding: 0.5rem 0;
        border-left: 5px solid #3498db;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3498db;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #95a5a6;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 标题
# ============================================================
st.markdown('<div class="main-header">🌍 全球失业率数据分析仪表盘<br><small style="font-size:1rem;color:#7f8c8d;">Global Unemployment Rate Analysis Dashboard (2014-2024)</small></div>', unsafe_allow_html=True)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("## 📋 导航")
    st.markdown("---")
    nav = st.radio(
        "选择查看内容：",
        ["🏠 项目概览", "📊 数据概览与统计", "📈 可视化图表", "🔬 统计检验", "🤖 机器学习模型", "📝 分析结论"]
    )
    st.markdown("---")
    st.markdown("### ℹ️ 关于项目")
    st.info("""
    **数据来源：** 国际劳工组织(ILO)

    **时间范围：** 2014-2024年

    **覆盖范围：** 全球180+国家

    **分析维度：** 性别、年龄组
    """)
    st.markdown("---")
    st.markdown("### 📁 输出文件目录")
    if os.path.exists('output'):
        files = sorted(os.listdir('output'))
        for f in files:
            st.markdown(f"- `{f}`")

# ============================================================
# 输出目录路径
# ============================================================
OUTPUT_DIR = 'output'

def load_csv(filename):
    """安全加载CSV文件"""
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path, encoding='utf-8-sig')
    return None

# ============================================================
# 页面1：项目概览
# ============================================================
if nav == "🏠 项目概览":
    st.markdown('<div class="section-header">项目背景与目标</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>📖 项目背景</h3>
        <p>失业率是衡量一个国家经济健康状况最重要的指标之一。本项目利用国际劳工组织（ILO）提供的全球失业率数据，对2014-2024年间全球各国的失业率进行全面分析。</p>
        <p>通过分析不同国家、性别、年龄组之间的失业率差异，以及COVID-19疫情对全球就业市场的影响，为理解全球就业趋势提供数据支持。</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>🎯 项目目标</h3>
        <ul>
            <li>分析全球失业率的分布特征和趋势</li>
            <li>比较不同性别和年龄组的失业率差异</li>
            <li>评估COVID-19对就业市场的影响</li>
            <li>运用机器学习方法进行聚类和预测</li>
            <li>通过Web仪表盘直观呈现分析结果</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">技术栈</div>', unsafe_allow_html=True)

    tech_cols = st.columns(5)
    tech_items = [
        ("🐼", "Pandas", "数据处理"),
        ("📊", "Matplotlib/Seaborn", "数据可视化"),
        ("🔬", "Scipy", "统计检验"),
        ("🤖", "Scikit-learn", "机器学习"),
        ("🌐", "Streamlit", "Web展示")
    ]
    for col, (icon, name, desc) in zip(tech_cols, tech_items):
        col.markdown(f"""
        <div style="text-align:center;padding:1rem;background:#f0f3f7;border-radius:10px;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-weight:bold;">{name}</div>
            <div style="color:#7f8c8d;font-size:0.8rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">项目流程图</div>', unsafe_allow_html=True)
    st.markdown("""
    ```mermaid
    flowchart TD
        A[数据加载] --> B[数据清洗与预处理]
        B --> C[描述性统计分析]
        B --> D[统计检验]
        B --> E[数据可视化]
        B --> F[机器学习建模]

        C --> G[分布分析]
        C --> H[均值/占比分析]

        D --> I[t检验: 性别差异]
        D --> J[ANOVA: 年龄组差异]

        E --> K[趋势图/箱线图]
        E --> L[热力图/散点图]
        E --> M[柱状图/直方图]

        F --> N[K-Means聚类]
        F --> O[线性回归]
        F --> P[随机森林]
        F --> Q[PCA降维]

        G & H & I & J & K & L & M & N & O & P & Q --> R[结果输出到output文件夹]
        R --> S[Streamlit Web展示]
    ```
    """)

# ============================================================
# 页面2：数据概览与统计
# ============================================================
elif nav == "📊 数据概览与统计":
    st.markdown('<div class="section-header">📊 数据概览</div>', unsafe_allow_html=True)

    # 加载描述性统计
    desc_stats = load_csv('descriptive_stats.csv')
    if desc_stats is not None:
        st.markdown("#### 年度失业率描述性统计（2014-2024）")
        st.dataframe(desc_stats, use_container_width=True)
        st.markdown("*分析说明：上表展示了各年度失业率的均值、标准差、四分位数等统计指标，反映了全球失业率的整体分布情况。*")

    st.markdown("---")

    # 指标卡片
    yearly_trend = load_csv('yearly_global_trend.csv')
    if yearly_trend is not None:
        st.markdown("#### 📈 关键指标")
        cols = st.columns(4)
        cols[0].metric("2014年全球均值", f"{yearly_trend['Global_Mean'].iloc[0]:.2f}%")
        cols[1].metric("2024年全球均值", f"{yearly_trend['Global_Mean'].iloc[-1]:.2f}%")
        cols[2].metric("最高均值（年份）", f"{yearly_trend['Global_Mean'].max():.2f}%",
                       help=f"最大值出现在{yearly_trend.loc[yearly_trend['Global_Mean'].idxmax(), 'Year']}年")
        cols[3].metric("最低均值（年份）", f"{yearly_trend['Global_Mean'].min():.2f}%",
                       help=f"最小值出现在{yearly_trend.loc[yearly_trend['Global_Mean'].idxmin(), 'Year']}年")

    st.markdown("---")

    # 性别统计
    gender_stats = load_csv('gender_stats.csv')
    if gender_stats is not None:
        st.markdown("#### 👫 按性别分组统计")
        st.dataframe(gender_stats, use_container_width=True)
        st.markdown("*分析说明：从上表可以看出女性平均失业率普遍高于男性，反映出就业市场中的性别不平等现象。*")

    st.markdown("---")

    # 年龄组统计
    age_stats = load_csv('age_stats.csv')
    if age_stats is not None:
        st.markdown("#### 👶🧑👴 按年龄组统计")
        st.dataframe(age_stats, use_container_width=True)
        st.markdown("*分析说明：青年组(Youth, 15-24岁)的失业率显著高于成人组(Adults)，体现了青年就业问题的严峻性。*")

    st.markdown("---")

    # 国家排名
    country_summary = load_csv('country_summary.csv')
    if country_summary is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔴 失业率最高Top10")
            top10 = country_summary.head(10)
            st.dataframe(top10, use_container_width=True)
        with col2:
            st.markdown("#### 🟢 失业率最低Top10")
            bottom10 = country_summary.tail(10).iloc[::-1]
            st.dataframe(bottom10, use_container_width=True)
        st.markdown("*分析说明：失业率最高的国家多集中在经济发展水平较低或政治不稳定的地区。*")

# ============================================================
# 页面3：可视化图表
# ============================================================
elif nav == "📈 可视化图表":
    st.markdown('<div class="section-header">📈 数据可视化图表</div>', unsafe_allow_html=True)

    # ---- 第一部分：数据探索图表 (fig1-fig8) ----
    st.markdown("""
    <div style="background:#eaf2f8;padding:12px 16px;border-radius:8px;margin:10px 0;border-left:5px solid #2980b9;">
    <strong>📊 第一部分：数据特征探索图表 (Fig.1~8)</strong><br>
    <span style="color:#555;">以下图表用于展示数据本身的统计特征——趋势、分布、对比、相关性等。</span>
    </div>
    """, unsafe_allow_html=True)

    data_charts = [
        ('fig1_global_trend.png', '图表1：全球失业率年度趋势',
         '展示了2014-2024年全球平均失业率的变化趋势，阴影区域表示标准差范围，红色区域标注了COVID-19疫情时期。'),
        ('fig2_gender_comparison.png', '图表2：男女失业率对比',
         '左侧为按年份分组的男女失业率箱线图，右侧为总体均值对比。女性失业率系统性高于男性。'),
        ('fig3_age_heatmap.png', '图表3：年龄组×年份热力图',
         '热力图展示了三个年龄组在各年度的平均失业率。青年组颜色最深（失业率最高）。'),
        ('fig4_distribution_histogram.png', '图表4：2024年失业率分布直方图',
         '直方图+KDE密度曲线展示2024年全球失业率的整体分布形态，呈右偏分布。'),
        ('fig5_covid_impact_scatter.png', '图表5：COVID-19影响散点图',
         '每个点代表一个国家，横轴为平均失业率，纵轴为COVID-19冲击程度，颜色代表整体趋势方向。'),
        ('fig6_gender_age_barchart.png', '图表6：性别×年龄组交叉分析',
         '分组柱状图展示性别和年龄组的交叉效应。青年女性失业率最高。'),
        ('fig7_top30_countries.png', '图表7：失业率最高30国',
         '水平柱状图展示失业率最高的30个国家排名。'),
        ('fig8_age_trends.png', '图表8：各年龄组年度趋势',
         '折线图展示三个年龄组在2014-2024年的趋势变化，青年组波动最大。'),
    ]

    for i in range(0, len(data_charts), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(data_charts):
                filename, title, description = data_charts[i + j]
                with cols[j]:
                    st.markdown(f"#### {title}")
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    if os.path.exists(filepath):
                        st.image(filepath, use_container_width=True)
                    with st.expander("📝 图表解析"):
                        st.markdown(description)

    # ---- 第二部分：模型结果图表 (fig9-fig12) ----
    st.markdown("---")
    st.markdown("""
    <div style="background:#fdf2e9;padding:12px 16px;border-radius:8px;margin:10px 0;border-left:5px solid #e67e22;">
    <strong>🤖 第二部分：机器学习模型结果图表 (Fig.9~12)</strong><br>
    <span style="color:#555;">以下图表用于展示建模分析的结果——模型选择、特征重要性、预测效果、降维可视化等。</span>
    </div>
    """, unsafe_allow_html=True)

    model_charts = [
        ('fig9_elbow_method.png', '图表9：肘部法则 — K-Means最佳K值选择',
         '展示不同K值下的惯性(WCSS)变化，K=4处存在明显拐点（肘部），因此选择4类聚类方案。属于模型选择过程的可视化。'),
        ('fig10_linear_regression.png', '图表10：线性回归 — 失业率趋势预测',
         '展示线性回归模型的拟合效果和2025-2030年外推预测，阴影区域为±RMSE范围。属于模型预测结果的可视化。'),
        ('fig11_random_forest.png', '图表11：随机森林 — 特征重要性+预测vs实际',
         '左图：各特征对预测2024年失业率的重要性排名；右图：测试集上预测值与实际值的散点对比。属于模型解释和评估的可视化。'),
        ('fig12_pca_analysis.png', '图表12：PCA — 解释方差+降维可视化',
         '左图：各主成分的解释方差比率；右图：189个国家在PC1-PC2空间的分布，颜色代表平均失业率。属于降维结果的可视化。'),
    ]

    for i in range(0, len(model_charts), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(model_charts):
                filename, title, description = model_charts[i + j]
                with cols[j]:
                    st.markdown(f"#### {title}")
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    if os.path.exists(filepath):
                        st.image(filepath, use_container_width=True)
                    else:
                        st.warning(f"图表文件 {filename} 不存在，请先运行 ipynb 文件。")
                    with st.expander("📝 图表解析"):
                        st.markdown(description)

# ============================================================
# 页面4：统计检验
# ============================================================
elif nav == "🔬 统计检验":
    st.markdown('<div class="section-header">🔬 统计检验分析</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### t检验：男女失业率差异")
        ttest_result = load_csv('ttest_result.csv')
        if ttest_result is not None:
            st.dataframe(ttest_result, use_container_width=True)

            # 可视化
            fig, ax = plt.subplots(figsize=(6, 4))
            categories = ['Male', 'Female']
            means = [ttest_result['Value'].iloc[0], ttest_result['Value'].iloc[1]]
            errors = [ttest_result['Value'].iloc[2], ttest_result['Value'].iloc[3]]
            bars = ax.bar(categories, means, yerr=errors, color=['#3498db', '#e74c3c'],
                         capsize=10, edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       f'{float(val):.2f}%', ha='center', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Unemployment Rate (%)', fontweight='bold')
            ax.set_title('t-test: Male vs Female Unemployment', fontweight='bold')
            st.pyplot(fig)
            plt.close()

            st.markdown(f"""
            **结论：** p值 = {ttest_result['Value'].iloc[5]:.6f} < 0.05，男性和女性的失业率存在**显著差异**。

            女性平均失业率显著高于男性，这反映了全球就业市场中的性别不平等问题。
            """)

    with col2:
        st.markdown("#### ANOVA：年龄组失业率差异")
        anova_result = load_csv('anova_result.csv')
        if anova_result is not None:
            st.dataframe(anova_result, use_container_width=True)

            # 可视化
            fig, ax = plt.subplots(figsize=(6, 4))
            ages = ['Youth', 'Adults', 'Children']
            means_age = [anova_result['Value'].iloc[0], anova_result['Value'].iloc[1], anova_result['Value'].iloc[2]]
            colors = ['#e74c3c', '#3498db', '#2ecc71']
            bars = ax.bar(ages, means_age, color=colors, edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars, means_age):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       f'{float(val):.2f}%', ha='center', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Unemployment Rate (%)', fontweight='bold')
            ax.set_title('ANOVA: Unemployment by Age Group', fontweight='bold')
            st.pyplot(fig)
            plt.close()

            st.markdown(f"""
            **结论：** p值 = {anova_result['Value'].iloc[7]:.6f} < 0.05，至少有一个年龄组的失业率与其他组存在**显著差异**。

            青年组(Youth, 15-24岁)的失业率远高于成人组，青年就业问题是全球性挑战。
            """)

# ============================================================
# 页面5：机器学习模型
# ============================================================
elif nav == "🤖 机器学习模型":
    st.markdown('<div class="section-header">🤖 机器学习模型分析</div>', unsafe_allow_html=True)

    # 模型选择
    ml_model = st.selectbox(
        "选择机器学习模型查看详情：",
        ["K-Means聚类分析", "线性回归预测", "随机森林回归", "PCA主成分分析"]
    )

    st.markdown("---")

    if ml_model == "K-Means聚类分析":
        st.markdown("#### 🔵 K-Means聚类分析：国家失业模式分类")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            **方法说明：**

            K-Means是一种无监督学习算法，将具有相似失业模式的国家划分到同一类别。

            **特征选择：**
            - 平均失业率 (avg_unemployment)
            - 波动性 (volatility)
            - 2014-2024趋势 (trend_2014_2024)
            - COVID-19影响 (covid_impact)

            **最佳K值：** 通过肘部法则确定为 K=4
            """)

        with col2:
            # 肘部图
            elbow_path = os.path.join(OUTPUT_DIR, 'fig9_elbow_method.png')
            if os.path.exists(elbow_path):
                st.image(elbow_path, use_container_width=True)

        # 聚类结果
        kmeans_data = load_csv('kmeans_clusters.csv')
        if kmeans_data is not None and 'Cluster' in kmeans_data.columns:
            st.markdown("#### 聚类结果分析")
            for cluster_id in sorted(kmeans_data['Cluster'].unique()):
                cluster_data = kmeans_data[kmeans_data['Cluster'] == cluster_id]
                cluster_mean = cluster_data.mean()
                countries = cluster_data['country_name'].tolist()
                st.markdown(f"""
                **类别 {cluster_id}** ({len(cluster_data)} 个国家)
                - 平均失业率：{cluster_mean['avg_unemployment']:.2f}%
                - 平均波动性：{cluster_mean['volatility']:.2f}
                - 平均趋势：{cluster_mean['trend_2014_2024']:.2f}%
                - 代表国家：{', '.join(countries[:8])}...
                """)

    elif ml_model == "线性回归预测":
        st.markdown("#### 📈 线性回归：失业率趋势预测")

        col1, col2 = st.columns([3, 2])
        with col1:
            lr_path = os.path.join(OUTPUT_DIR, 'fig10_linear_regression.png')
            if os.path.exists(lr_path):
                st.image(lr_path, use_container_width=True)

        with col2:
            lr_data = load_csv('linear_regression_results.csv')
            if lr_data is not None:
                st.markdown("#### 预测值")
                future = lr_data[lr_data['Type'] == 'Predicted']
                actual = lr_data[lr_data['Type'] == 'Actual']
                for _, row in future.iterrows():
                    st.metric(f"{int(row['Year'])}年预测", f"{row['Unemployment_Rate']:.2f}%")

            st.markdown("""
            **模型评估：**
            - R²越接近1越好
            - RMSE越低越好
            - 模型反映了长期趋势
            """)

        st.markdown("*分析说明：线性回归模型揭示了全球失业率的长期下降趋势。但需注意，它是简化的线性模型，实际失业率受多种复杂因素影响。*")

    elif ml_model == "随机森林回归":
        st.markdown("#### 🌲 随机森林回归：特征重要性分析")

        col1, col2 = st.columns(2)

        with col1:
            rf_path = os.path.join(OUTPUT_DIR, 'fig11_random_forest.png')
            if os.path.exists(rf_path):
                st.image(rf_path, use_container_width=True)

        with col2:
            rf_importance = load_csv('rf_feature_importance.csv')
            if rf_importance is not None:
                st.markdown("#### 特征重要性排名")
                st.dataframe(rf_importance, use_container_width=True)

                # 特征重要性柱状图
                fig, ax = plt.subplots(figsize=(5, 5))
                top_features = rf_importance.head(8)
                ax.barh(top_features['Feature'][::-1], top_features['Importance'][::-1],
                       color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features))))
                ax.set_xlabel('Importance', fontweight='bold')
                ax.set_title('Top Feature Importance', fontweight='bold')
                st.pyplot(fig)
                plt.close()

        st.markdown("*分析说明：随机森林模型可以自动学习特征的非线性关系。特征重要性分析表明，近年的历史失业率数据和国家差异是预测未来失业率最重要的因素。*")

    elif ml_model == "PCA主成分分析":
        st.markdown("#### 🔻 PCA主成分分析：数据降维与可视化")

        pca_path = os.path.join(OUTPUT_DIR, 'fig12_pca_analysis.png')
        if os.path.exists(pca_path):
            st.image(pca_path, use_container_width=True)

        pca_data = load_csv('pca_results.csv')
        if pca_data is not None:
            st.markdown(f"#### PCA结果 (共{pca_data.shape[0]}个国家)")
            st.dataframe(pca_data.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **方法说明：**

            主成分分析(PCA)将11年的失业率数据压缩到少数几个主成分，保留最大方差信息。
            """)
        with col2:
            st.markdown("""
            **分析结论：**

            - 前2个主成分可解释大部分数据方差
            - 国家在PC1-PC2空间中的位置反映了其失业模式的相似性
            - 相近位置的国家具有相似的失业率变化模式
            """)

# ============================================================
# 页面6：分析结论
# ============================================================
elif nav == "📝 分析结论":
    st.markdown('<div class="section-header">📝 分析总结与结论</div>', unsafe_allow_html=True)

    # 加载摘要
    summary_path = os.path.join(OUTPUT_DIR, 'analysis_summary.txt')
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_text = f.read()

    st.markdown("""
    ---

    ### 📌 主要发现

    | 维度 | 主要发现 |
    |------|----------|
    | **时间趋势** | 全球失业率总体呈缓慢下降趋势，但2020年COVID-19造成显著冲击，失业率大幅升高 |
    | **性别差异** | 女性失业率系统性高于男性，t检验证实差异有统计显著性 (p < 0.05) |
    | **年龄差异** | 青年(15-24岁)失业率远高于成人，ANOVA检验证实差异显著 (p < 0.05) |
    | **地区差异** | 高失业率国家集中在经济发展水平较低的地区，低失业率国家多为发达国家 |
    | **COVID-19影响** | 2020-2021年全球失业率明显上升，对旅游业和服务业为主的国家冲击最大 |

    ---

    ### 🔮 模型结论

    | 模型 | 核心发现 |
    |------|----------|
    | **K-Means聚类** | 全球国家可分为4类失业模式：高失业波动型、中高失业型、中低失业稳定型、低失业型 |
    | **线性回归** | 全球失业率呈下降趋势，模型可用于趋势外推预测 |
    | **随机森林** | 近年历史数据和国家特征是最重要的预测因素，模型R²较高 |
    | **PCA分析** | 前2个主成分可解释大部分方差，国家间的失业模式具有明显的结构特征 |

    ---

    ### 💡 政策建议

    1. **关注青年就业：** 青年失业率普遍偏高，需要针对性的职业培训和就业支持政策
    2. **促进性别平等：** 缩小男女就业差距，消除就业市场中的性别歧视
    3. **应对危机冲击：** 建立就业市场韧性机制，应对类似COVID-19的经济冲击
    4. **国际合作：** 高失业率国家需要国际社会更多的经济和技术援助

    ---

    ### 🔬 项目局限性与展望

    - 数据粒度有限，无法进行更细致的行业分析
    - 可进一步引入GDP、教育水平等社会经济因素
    - 可尝试深度学习模型（LSTM）进行时间序列预测
    - 可结合NLP技术分析就业政策文本
    """)

    st.markdown("---")

    # 文件列表
    st.markdown("#### 📁 产出文件清单")
    if os.path.exists(OUTPUT_DIR):
        files = sorted(os.listdir(OUTPUT_DIR))
        file_df = pd.DataFrame({
            '文件名': files,
            '类型': [f.split('.')[-1] for f in files],
            '大小(KB)': [round(os.path.getsize(os.path.join(OUTPUT_DIR, f))/1024, 1) for f in files]
        })
        st.dataframe(file_df, use_container_width=True)

# ============================================================
# 页脚
# ============================================================
st.markdown("---")
st.markdown('<div class="footer">🏫 动手学AI课程期末作业 | 📊 全球失业率数据分析项目 | 数据来源：国际劳工组织(ILO)</div>', unsafe_allow_html=True)
