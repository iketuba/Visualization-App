import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("データ可視化アプリ")

uploaded_file = st.file_uploader("ファイルをアップロードして下さい")

if uploaded_file is not None:
    # データの基本的な情報
    st.markdown("## データの基本的な情報")

    dataframe = pd.read_csv(uploaded_file)

    st.markdown("""### データの表示、サイズ""")
    st.write(dataframe.head())
    st.write(f"{dataframe.shape[0]}行 {dataframe.shape[1]}列")

    st.markdown("""### 欠損値""")
    null_count = dataframe.isnull().sum()
    null_ratio = round(dataframe.isnull().sum() / len(dataframe), 2)
    null_dataframe = pd.concat([null_count, null_ratio], axis=1)
    null_dataframe.columns = ["欠損数", "欠損割合"]
    st.write(null_dataframe)

    st.markdown("""### データ型""")
    type_dataframe = pd.DataFrame(dataframe.dtypes, columns=["データ型"])
    st.write(type_dataframe.astype(str))

    # データの可視化
    st.markdown("## データの可視化")

    st.markdown("### 1変数の可視化")
    x = list(dataframe.columns)
    x.insert(0, "未選択")
    option_feature_1 = st.selectbox("どの変数を可視化しますか?",tuple(x), key=1)

    if option_feature_1 != "未選択":
        if dataframe[option_feature_1].dtype in ["int64", "float64"]:
            option_method_1 = st.selectbox("グラフの種類を選んで下さい",("未選択", "countplot", "distplot", "boxplot"), key=2)
            if option_method_1 == "countplot":
                fig = plt.figure(figsize=(12, 8))
                sns.countplot(data=dataframe, x=option_feature_1)
                st.pyplot(fig)

            elif option_method_1 == "distplot":
                fig = plt.figure(figsize=(12, 8))
                sns.distplot(dataframe[option_feature_1])
                st.pyplot(fig)

            elif option_method_1 == "boxplot":
                fig = plt.figure(figsize=(12, 8))
                sns.boxplot(dataframe[option_feature_1])
                st.pyplot(fig)

        elif dataframe[option_feature_1].dtype in ["object"]:
            option_method_2 = st.selectbox("グラフの種類を選んで下さい", ("未選択", "countplot", "countplot(多い順)", "countplot(多い順に10項目まで)"), key=3)
            if option_method_2 == "countplot":
                fig = plt.figure(figsize=(12, 8))
                sns.countplot(data=dataframe, x=option_feature_1)
                plt.xticks(rotation=90)
                st.pyplot(fig)

            elif option_method_2 == "countplot(多い順)":
                order = dataframe[option_feature_1].value_counts().sort_values(ascending=False).index                
                fig = plt.figure(figsize=(12, 8))
                sns.countplot(data=dataframe, x=option_feature_1, order=order)
                plt.xticks(rotation=90)
                st.pyplot(fig)       

            elif option_method_2 == "countplot(多い順に10項目まで)":
                order = dataframe[option_feature_1].value_counts().sort_values(ascending=False).index
                order_top10 = order[:10]                
                fig = plt.figure(figsize=(12, 8))
                dataframe_top10 = dataframe[dataframe[option_feature_1].isin(order_top10)]
                sns.countplot(data=dataframe_top10, x=option_feature_1, order=order_top10)
                plt.xticks(rotation=90)
                st.pyplot(fig)                           

    st.markdown("### 目的変数との関係")
    x = list(dataframe.columns)
    x.insert(0, "未選択")
    option_target = st.selectbox("目的変数を選択して下さい", tuple(x), key=4)

    if option_target != "未選択":
        option_feature_2 = st.selectbox("どの変数との関係を可視化しますか?",tuple(x), key=5)
        if option_feature_2 != "未選択":
            option_problem = st.selectbox("回帰問題か分類問題か選択して下さい", ("未選択", "回帰", "分類"), key=6)
            if option_problem == "分類":
                if dataframe[option_feature_2].dtype in ["int64", "float64"]:
                    option_method_3 = st.selectbox("グラフの種類を選んで下さい",("未選択", "distplot", "boxplot", "countplot"), key=7)                     

                    if option_method_3 == "distplot":
                        target_count = dataframe[option_target].nunique()
                        fig, ax = plt.subplots(1, target_count, figsize=(12, 8))
                        for i in range(target_count):
                            target_value = dataframe[option_target].unique()[i]
                            sns.distplot(dataframe[dataframe[option_target] == target_value][option_feature_2], ax=ax[i])
                            ax[i].set_title(target_value)
                        st.pyplot(fig)

                    elif option_method_3 == "boxplot":
                        fig = plt.figure(figsize=(12, 8))
                        sns.boxplot(data=dataframe, x=option_target, y=option_feature_2)
                        st.pyplot(fig)

                    elif option_method_3 == "countplot":
                        target_count = dataframe[option_target].nunique()
                        fig, ax = plt.subplots(1, target_count, figsize=(12, 8))
                        for i in range(target_count):
                            target_value = dataframe[option_target].unique()[i]
                            sns.countplot(data=dataframe[dataframe[option_target] == target_value], x=option_feature_2, ax=ax[i])
                            ax[i].set_title(target_value)
                        st.pyplot(fig)                                   

                elif dataframe[option_feature_2].dtype in ["object"]:
                    option_method_4 = st.selectbox("グラフの種類を選んで下さい",("未選択", "countplot"), key=8)
                    if option_method_4 == "countplot":
                        target_count = dataframe[option_target].nunique()
                        fig, ax = plt.subplots(1, target_count, figsize=(12, 8))
                        for i in range(target_count):
                            target_value = dataframe[option_target].unique()[i]
                            sns.countplot(data=dataframe[dataframe[option_target] == target_value], x=option_feature_2, ax=ax[i])
                            ax[i].set_title(target_value)
                        st.pyplot(fig)

            elif option_problem == "回帰":
                if dataframe[option_feature_2].dtype in ["int64", "float64"]:
                    # 数値変数だがカテゴリ変数のような場合に、countplotが必要な場合があるかもしれない
                    option_method_5 = st.selectbox("グラフの種類を選んで下さい",("未選択", "regplot"), key=9)
                    if option_method_5 == "regplot":
                        fig = plt.figure(figsize=(12, 8))
                        sns.regplot(data=dataframe, x=option_feature_2, y=option_target)
                        st.pyplot(fig)

                elif dataframe[option_feature_2].dtype in ["object"]:
                    option_method_6 = st.selectbox("グラフの種類を選んで下さい",("未選択", "distplot", "distplot(10項目のみ)", "boxplot", "boxplot(10項目のみ)"), key=10)
                    if option_method_6 == "distplot":
                        explain_count = dataframe[option_feature_2].nunique()
                        fig, ax = plt.subplots(1, explain_count, figsize=(12, 8))
                        for i in range(explain_count):
                            explain_value = dataframe[option_feature_2].unique()[i]
                            sns.distplot(dataframe[dataframe[option_feature_2] == explain_value][option_target], ax=ax[i])
                            ax[i].set_title(explain_value)
                        st.pyplot(fig)

                    elif option_method_6 == "distplot(10項目のみ)":
                        explain_value_top10 = dataframe[option_feature_2].value_counts().sort_values(ascending=False).index[:10]
                        dataframe_top10 = dataframe[dataframe[option_feature_2].isin(explain_value_top10)]

                        explain_count = dataframe_top10[option_feature_2].nunique()
                        fig, ax = plt.subplots(1, explain_count, figsize=(12, 8))
                        for i in range(explain_count):
                            explain_value = dataframe_top10[option_feature_2].unique()[i]
                            sns.distplot(dataframe_top10[dataframe_top10[option_feature_2] == explain_value]["SalePrice"], ax=ax[i])
                            ax[i].set_title(explain_value)
                        st.pyplot(fig)                   

                    elif option_method_6 == "boxplot":
                        fig = plt.figure(figsize=(12, 8))
                        sns.boxplot(data=dataframe, y=option_target, x=option_feature_2)
                        st.pyplot(fig)

                    elif option_method_6 == "boxplot(10項目のみ)":
                        explain_value_top10 = dataframe[option_feature_2].value_counts().sort_values(ascending=False).index[:10]
                        dataframe_top10 = dataframe[dataframe[option_feature_2].isin(explain_value_top10)]                        

                        fig = plt.figure(figsize=(12, 8))
                        sns.boxplot(data=dataframe_top10, y=option_target, x=option_feature_2)
                        st.pyplot(fig)
                     
