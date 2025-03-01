"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import streamlit.components.v1 as components  # html extensions
# st.set_page.config(layout='wide', initial_sidebar_state='expanded')
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import base64
from streamlit_shap import st_shap
import shap
import lime

from streamlit_javascript import st_javascript

# Data handling dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest

# App declaration
def main():
    # st.sidebar.markdown('side')
    st.markdown(
        """
        <style>
        .reportview-container {
        background: url('resources/imgs/sample.jpg')
        }
        .sidebar .sidebar-content {
        background: url('resources/imgs/sample.jpg')
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    with st.sidebar:
        #from PIL import Image
        #image2 = Image.open('resources/imgs/fraud.jpg')
        #st.image(image2, caption='Data Analytics')

        page_selection = option_menu(
            menu_title=None,
            options=["Overview", "Step 3: Output", "Step 2: Model", "Step 1: Input"],
            icons=['file-earmark-text', 'graph-up', 'robot', 'file-earmark-spreadsheet'],
            menu_icon='cast',
            default_index=0,
            # orientation='horizontal',
            styles={"container": {'padding': '0!important', 'background_color': 'red'},
                    'icon': {'color': 'red', 'font-size': '18px'},
                    'nav-link': {
                        'font-size': '15px',
                        'text-align': 'left',
                        'margin': '0px',
                        '--hover-color': '#4BAAFF',
                    },
                    'nav-link-selected': {'background-color': '#6187D2'},
                    }
        )
        st.info('This algorithm predicts which financial statements are anomalous and explains why. '
                'An explanation is given about the data that is used (input), '
                'the algorithm that is used (model) and about the predictions of the algorithm (output)', icon="ℹ️")
    # page_options = ["Recommender System","Movie Facts","Exploratory Data Analysis","About"]




    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    # page_selection = st.sidebar.radio("Choose Option", page_options)
    if page_selection == "Overview":
        # Header contents
        st.title('Financial Fraud Detection Application')
        st.write('### Detecting and Preventing Fraud with Financial Statements')
        st.markdown("This app is designed to flag potentially fraudulent financial statements submitted by companies with an insurance guarantee. By using a statistical (or other) model to predict and flag fraudulent statements, the app provides credit underwriters with an extra variable to consider when assessing credit risk. "
                    "The app analyzes the predicted fraud across different buckets such as industry, year, and financial type, providing insights into the appropriateness of the data for modeling purposes. It also generates a fraud indicator (or probability) that can be incorporated into the credit risk model. "
                    )

        st.image('resources/imgs/ov2.jpg')
        st.write("### Why use machine learning for fraud detection?")
        st.markdown("- **Fraud detection machine learning models are more effective than humans**")
        st.markdown("- **ML handles overload well**")
        st.markdown("- **ML beats traditional fraud detection systems**")






    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Step 2: Model":
        # Header Contents
        st.write("# Anomaly Detection Models")

        if st.checkbox("Algorithm Explanation"):
            filters = ["Benford's Law","Isolation Forest", "Local Outlier Factor"]
            filter_selection = st.selectbox("**Algorithms**", filters)
            if filter_selection == "Benford's Law":
                st.write("## Benford's Law Analysis")
                st.markdown("Benford's Law is a very simple statistical law that can be used to detect the probability of fraud in any given dataset.")
                st.markdown("In some cases, big problems can be solved in a very simple way. It doesn't require high-level algorithms, coding, models, etc.")
                st.markdown("### What is Benford's Law?")
                st.image('resources/imgs/ben.png')
                st.markdown("Visual representation of the Benford's Law")
                st.markdown("Statistically, Benford's law is known as the first-digit law, which states that in many naturally occurring datasets, the first digit of a number is usually small rather than large.")
                st.markdown("In particular, the law predicts that the first digit will appear about 30% of the time, while the digit 9 will appear less than 5% of the time.")
                st.markdown("### How can we use it to detect fraud?")
                st.markdown("If a dataset is significantly different from the expected distribution of the first digits, this may be an indication that it has been altered or fabricated.")
                st.markdown("For example, if a company's financial statements show an unusually high proportion of numbers starting with the digit 9, it could be a red flag for fraudulent activity.")
                st.markdown("### A visual representation of Benford's Law compared to example financial statements")
                st.image('resources/imgs/ben2.png')
                st.markdown("The above graph shows the percentage deviation from Benfod's law. Looking at the graph, it can easily be determined if the dataset is deviating from Benford's Law. The error margins were set at 10% in the graph above.")


            if filter_selection == "Isolation Forest":
                st.write("## Isolation Forest Algorithm")
                st.markdown("Isolation Forest is an unsupervised machine learning algorithm for anomaly detection. "
                            "As the name implies, Isolation Forest is an ensemble method (similar to random forest). "
                            "In other words, it use the average of the predictions by several decision trees when assigning the final anomaly score to a given data point. "
                            "Unlike other anomaly detection algorithms, which first define what’s “normal” and then report anything else as anomalous, "
                            "Isolation Forest attempts to isolate anomalous data points from the get go.")
                st.subheader("The algorithm")
                st.markdown("Suppose we had the following data points:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image('resources/imgs/is1.png', use_column_width=True)
                    st.markdown(
                        "The isolation forest algorithm selects a random dimension (in this case, the dimension associated with the x axis) and randomly splits the data along that dimension.")
                with col2:
                    st.image('resources/imgs/is2.png', use_column_width=True)
                    st.markdown(
                        "The two resulting subspaces define their own sub tree. In this example, the cut happens to separate a lone point from the remainder of the dataset. The first level of the resulting binary tree consists of two nodes, one which will consist of the subtree of points to the left of the initial cut and the other representing the single point on the righ")
                with col3:
                    st.image('resources/imgs/is3.png', use_column_width=True)
                    st.markdown(
                        "It’s important to note, the other trees in the ensemble will select different starting splits. In the following example, the first split doesn’t isolate the outlier.")
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.image('resources/imgs/is4.png', use_column_width=True)
                    st.markdown(
                        "We end up with a tree consisting of two nodes, one that contains the points to the left of the line and the other representing the points on the right side of the line.")
                with col5:
                    st.image('resources/imgs/is5.png', use_column_width=True)
                    st.markdown(
                        "The process is repeated until every leaf of the tree represents a single data point from the dataset. In our example, the second iteration manages to isolate the outlier.")
                with col6:
                    st.image('resources/imgs/is6.png', use_column_width=True)
                    st.markdown("After this step, the tree would look as follows:")

                col7, col8, col9 = st.columns(3)
                with col7:
                    st.image('resources/imgs/is7.png', use_column_width=True)
                    st.markdown(
                        "Remember that a split can occur along the other dimension as is the case for this 3rd decision tree.")
                with col8:
                    st.image('resources/imgs/is8.png', use_column_width=True)
                    st.markdown(
                        "On average, an anomalous data point is going to be isolated in a bounding box at a smaller tree depth than other points.")
                with col9:
                    st.image('resources/imgs/is9.png', use_column_width=True)
                    st.markdown(
                        "When performing inference using a trained Isolation Forest model the final anomaly score is reported as the average across scores reported by each individual decision tree.")

            if filter_selection == "Local Outlier Factor":
                st.write("## Local Outlier Factor Algorithm")
                st.subheader("What is LOF?")
                st.markdown(
                    "Local outlier factor (LOF) is an algorithm that identifies the outliers present in the dataset. But what does the **local outlier** mean?")
                st.markdown(
                    "When a point is considered as an outlier based on its local neighborhood, it is a local outlier. LOF will identify an outlier considering the density of the neighborhood. LOF performs well when the density of the data is not the same throughout the dataset.")
                st.markdown("To understand LOF, it is important to have an understanding of the following concepts:")
                st.markdown("- Distance and K-neighbors")
                st.markdown("- Reachability distance (RD)")
                st.markdown("- Local reachability density (LRD)")
                st.markdown("- Local Outlier Factor (LOF)")
                st.subheader("Distance and K-neighbors")
                st.markdown(
                    "K-distance is the distance between the point, and it’s Kᵗʰ nearest neighbor. K-neighbors denoted by Nₖ(A) includes a set of points that lie in or on the circle of radius K-distance. K-neighbors can be more than or equal to the value of K. How’s this possible?")
                st.markdown("Below an example is given. Let’s say we have four points A, B, C, and D (shown below).")
                st.image('resources/imgs/lo1.png', width=500,
                         caption='K-distance of A with K=2')
                st.markdown(
                    "If K=2, K-neighbors of A will be C, B, and D. Here, the value of K=2 but the ||N₂(A)|| = 3. Therefore, ||Nₖ(point)|| will always be greater than or equal to K.")
                st.subheader("Reachability distance (RD)")
                st.latex(r'''
                            RD( X_{i}   , X_{J} )  =  max(K  -  distance(X_{i}), distance(X_{i}, X_{J}))     
                            ''')
                st.markdown(
                    "It is defined as the maximum of K-distance of Xj and the distance between Xi and Xj. The distance measure is problem-specific (Euclidean, Manhattan, etc.)")
                st.image('resources/imgs/lo2.png', width=500,
                         caption='Illustration of reachability distance with K=2')
                st.markdown(
                    "In layman terms, if a point Xi lies within the K-neighbors of Xj, the reachability distance will be K-distance of Xj (blue line), else reachability distance will be the distance between Xi and Xj (orange line).")
                st.subheader("Local reachability density (LRD)")
                st.latex(r'''
                           LRD_{k}(A) =  \frac{1}{ \sum x_{j} \epsilon N_{k} \frac{RD(A,X_{j})}{ \| N_{K}(A) \| }} 
                            ''')
                st.markdown(
                    "LRD is inverse of the average reachability distance of A from its neighbors. Intuitively according to LRD formula, more the average reachability distance (i.e., neighbors are far from the point), less density of points are present around a particular point. This tells how far a point is from the nearest cluster of points. :blue[Low values of LRD implies that the closest cluster is far from the point.]")

                st.subheader("Local Outlier Factor (LOF)")
                st.markdown(
                    "LRD of each point is used to compare with the average LRD of its K neighbors. LOF is the ratio of the average LRD of the K neighbors of A to the LRD of A.")
                st.markdown(
                    "Intuitively, if the point is not an outlier (inlier), the ratio of average LRD of neighbors is approximately equal to the LRD of a point (because the density of a point and its neighbors are roughly equal). In that case, LOF is nearly equal to 1. On the other hand, if the point is an outlier, the LRD of a point is less than the average LRD of neighbors. Then LOF value will be high.")
                st.markdown(
                    " :red[Generally, if LOF> 1, it is considered as an outlier], but that is not always true. Let’s say we know that we only have one outlier in the data, then we take the maximum LOF value among all the LOF values, and the point corresponding to the maximum LOF value will be considered as an outlier.")
                col11, col22 = st.columns(2)
                with col11:
                    st.markdown("### :red[**LOF >> 1 (Anomaly)**]")
                with col22:
                    st.markdown("### :green[**LOF ~= 1 (Normal)**]")

                st.subheader("Advantages of LOF")
                st.markdown(
                    "A point will be considered as an outlier if it is at a small distance to the extremely dense cluster. The global approach may not consider that point as an outlier. But the LOF can effectively identify the local outliers.")
                st.subheader("Disavabtages of LOF")
                st.markdown(
                    "Since LOF is a ratio, it is tough to interpret. There is no specific threshold value above which a point is defined as an outlier. The identification of an outlier is dependent on the problem and the user.")


    # -------------------------------------------------------------
    if page_selection == "Step 3: Output":
        # Header Contents
        st.write("# Output of the Model")
        sys = st.radio("**Select an algorithm**", ('Statistical Analysis','Isolation Forest', 'Local Outlier Factor'))

        # Perform top-10 movie recommendation generation
        if sys == 'Statistical Analysis':
            if st.button("Detect"):
                try:
                    with st.spinner("Fitting Benford's Law ..."):
                        st.markdown("#### Analysis on Income Statement")
                        st.markdown("**Construction Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isconrev.png')
                            st.image(image2, caption='Revenue')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/iscongro.png')
                            st.image(image2, caption='Gross Revenue')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isconebi.png')
                            st.image(image2, caption='EBIT')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isconnet.png')
                            st.image(image2, caption='NetProfitAfterTax')

                        st.markdown("**UNKNOWN Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isunkrev.png')
                            st.image(image2, caption='Revenue')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isunkgro.png')
                            st.image(image2, caption='Gross Revenue')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isunkebi.png')
                            st.image(image2, caption='EBIT')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isunknet.png')
                            st.image(image2, caption='NetProfitAfterTax')

                        st.markdown("**Logistic Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/iclogrev.png')
                            st.image(image2, caption='Revenue')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isloggros.png')
                            st.image(image2, caption='Gross Revenue')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/iclogebi.png')
                            st.image(image2, caption='EBIT')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/iclognetprof.png')
                            st.image(image2, caption='NetProfitAfterTax')

                        st.markdown("**Energy Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isenerev.png')
                            st.image(image2, caption='Revenue')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isenegro.png')
                            st.image(image2, caption='Gross Revenue')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/iseneebi.png')
                            st.image(image2, caption='EBIT')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isenenet.png')
                            st.image(image2, caption='NetProfitAfterTax')


                        st.markdown("#### Analysis on Balance Sheet")
                        st.markdown("**Construction Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bstotalequ.png')
                            st.image(image2, caption='Total Equity')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bscontotalass.png')
                            st.image(image2, caption='Total Assets')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bscontotallia.png')
                            st.image(image2, caption='Total Liabilities')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bscontotalnet.png')
                            st.image(image2, caption='Networth')

                        st.markdown("**UNKNOWN Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsunktotalass.png')
                            st.image(image2, caption='Total Assets')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsunktotallia.png')
                            st.image(image2, caption='Total liability')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsunktotalequ.png')
                            st.image(image2, caption='Total Equity')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsunknetw.png')
                            st.image(image2, caption='Networth')

                        st.markdown("**Logistic Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bslogtotalequ.png')
                            st.image(image2, caption='Total Equity')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bslogtotalass.png')
                            st.image(image2, caption='Total Assets')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bslogtotallia.png')
                            st.image(image2, caption='Total Liabilities')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bslognetw.png')
                            st.image(image2, caption='Networth')

                        st.markdown("**Energy Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsenetotalass.png')
                            st.image(image2, caption='Total Assets')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsenetotallia.png')
                            st.image(image2, caption='Total liability')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsenetotalequ.png')
                            st.image(image2, caption='Total Equity')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsenenetw.png')
                            st.image(image2, caption='Networth')



                        st.markdown("#### Analysis on Cashflow Statement")
                        st.markdown("**Construction Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfconnetcf.png')
                            st.image(image2, caption='CFF_NetCFF')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfconcash.png')
                            st.image(image2, caption='CFF_CashAtStartOfYear')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfconcashend.png')
                            st.image(image2, caption='CFF_CashAtEndOfYear')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfconnetcfo.png')
                            st.image(image2, caption='CFO_NetCFO')


                        st.markdown("**UNKNOWN Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfunknetcf.png')
                            st.image(image2, caption='CFF_NetCFF')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfunkcash.png')
                            st.image(image2, caption='CFF_CashAtStartOfYear')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfunkcashend.png')
                            st.image(image2, caption='CFF_CashAtEndOfYear')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfunknetcfo.png')
                            st.image(image2, caption='CFO_NetCFO')

                        st.markdown("**Logistic Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cflognetcf.png')
                            st.image(image2, caption='CFF_NetCFF')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cflogcash.png')
                            st.image(image2, caption='CFF_CashAtStartOfYear')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cflogcashend.png')
                            st.image(image2, caption='CFF_CashAtEndOfYear')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cflogcashend.png')
                            st.image(image2, caption='CFO_NetCFO')

                        st.markdown("**Energy Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfenenetcf.png')
                            st.image(image2, caption='CFF_NetCFF')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfenecash.png')
                            st.image(image2, caption='CFF_CashAtStartOfYear')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfenecashend.png')
                            st.image(image2, caption='CFF_CashAtEndOfYear')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfenenetcfo.png')
                            st.image(image2, caption='CFO_NetCFO')

                except Exception as e:
                    st.write(e)
                    st.error("Oops! Looks like this algorithm doesn't work.\
                              We'll need to fix it!")


        if sys == 'Isolation Forest':
            if st.button("Detect"):
                try:
                    with st.spinner('Algorithm running...'):
                        df_new = pd.read_csv('resources/data/df_new.csv')

                        # Convert date column to datetime
                        #df1['FinancialsDate'] = pd.to_datetime(df1['FinancialsDate', 'Year', 'Month', 'Week', 'Day', 'Date','ReturnEquityRatio'])
                        df1 = df_new.drop(
                            ['FinancialsDate', 'Year', 'Month', 'Week', 'Day', 'Date', 'ReturnEquityRatio'], axis=1)


                        # Encode categorical columns using one-hot encoding
                        encoder = OneHotEncoder()
                        encoded_cat_columns = encoder.fit_transform(df1[['Financial_Type', 'Country', 'Industry']])
                        encoded_cat_columns_df = pd.DataFrame(encoded_cat_columns.toarray(),
                                                              columns=encoder.get_feature_names(
                                                                  ['Financial_Type', 'Country', 'Industry']))

                        # Combine encoded categorical columns with numerical columns
                        X = pd.concat(
                            [df1.drop(['Financial_Type', 'Country', 'Industry'], axis=1), encoded_cat_columns_df],
                            axis=1)

                        iforest = IsolationForest(max_samples='auto', bootstrap=False, n_jobs=-1, random_state=42)
                        iforest_ = iforest.fit(X)
                        y_pred = iforest_.predict(X)

                        y_score = iforest.decision_function(X)
                        neg_value_indices = np.where(y_score < 0)
                        arr = [ 492,  493,  512,  941,  943, 1588, 1852, 1853, 1854, 1855, 2140,
        2141, 2142, 2143, 2221, 2222, 2247, 2342, 2350, 2357, 2382, 2428,
        2429, 2492, 2533, 2565, 2602, 2626, 2632, 2698, 2699, 2750, 2751,
        2795, 2796, 2828, 2829, 2854, 2855, 2860, 2862, 2863, 2868, 2886,
        2887, 2888, 2889, 2890, 2891, 2893, 2894, 2895, 2896, 2897, 2898,
        2903, 2905, 2947, 2990, 2991, 2992, 2993, 3047, 3048, 3052, 3060,
        3061, 3064, 3066, 3067, 3082, 3109, 3137, 3156, 3159, 3172, 3182,
        3268, 3286, 3294, 3295, 3296, 3606, 3655, 3695, 3797, 3811, 3822,
        3823, 3985, 3986, 4016, 4038, 4066, 4075, 4076, 4077, 4103, 4104,
        4114, 4115, 4116, 4127, 4132, 4138, 4145, 4147, 4150, 4151, 4184,
        4185, 4214, 4215, 4216, 4229, 4230, 4231, 4241, 4303, 4304, 4313,
        4314, 4315, 4316, 4317, 4318, 4378, 4379, 4380, 4381, 4382, 4412,
        4413, 4420, 4421, 4434, 4452, 4453]

                        # Converting that array into a dataframe
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        outliers_df = df1.iloc[arr]
                        st.success("### Results are ready!")
                        st.markdown("### 138 possible fraudulent statements identified")
                        st.dataframe(outliers_df)  # Same as st.write(df)
                        # model here



                except Exception as e:
                    st.write(e)
                    st.error("Oops! Looks like this algorithm doesn't work.\
                                      We'll need to fix it!")

                df_final = outliers_df
        st.subheader('Isolation Forest Interpretability')
        col1, col2 = st.columns(2)
        if st.checkbox("Global Interpretability"):
            st.subheader("Global Machine Learning Interpretability")
            st.image('resources/imgs/ifglobal.png',use_column_width=True, caption = 'Summary Plot.')
            st.markdown("From this plot, the impact of a particular variable on anomaly detection is observed. Taking NCA_TotalLoansIssued, CL_InstalmentSaleLiabilty or CL_BankOverdraft as an example. The summary plot says that high values of that variables show anomalous observations while lower values are normal items.")

            st.image('resources/imgs/ifglobalbar.png', use_column_width=True, caption='Bar Plot.')
            st.markdown(
            "From the above, the variables CA_TradeAndOtherRecievables, NCA_TotalLoansIssued and VA_NCL_TotalEquityAndLiability_TotalEquity have the highest average SHAP value. Hence, they have the highest impact on determining the anomaly score..")

        if st.checkbox("Local Interpretability"):
            st.subheader("Local Machine Learning Interpretability")

            my_list = [ 492,  493,  512,  941,  943, 1588, 1852, 1853, 1854, 1855, 2140,
        2141, 2142, 2143, 2221, 2222, 2247, 2342, 2350, 2357, 2382, 2428,
        2429, 2492, 2533, 2565, 2602, 2626, 2632, 2698, 2699, 2750, 2751,
        2795, 2796, 2828, 2829, 2854, 2855, 2860, 2862, 2863, 2868, 2886,
        2887, 2888, 2889, 2890, 2891, 2893, 2894, 2895, 2896, 2897, 2898,
        2903, 2905, 2947, 2990, 2991, 2992, 2993, 3047, 3048, 3052, 3060,
        3061, 3064, 3066, 3067, 3082, 3109, 3137, 3156, 3159, 3172, 3182,
        3268, 3286, 3294, 3295, 3296, 3606, 3655, 3695, 3797, 3811, 3822,
        3823, 3985, 3986, 4016, 4038, 4066, 4075, 4076, 4077, 4103, 4104,
        4114, 4115, 4116, 4127, 4132, 4138, 4145, 4147, 4150, 4151, 4184,
        4185, 4214, 4215, 4216, 4229, 4230, 4231, 4241, 4303, 4304, 4313,
        4314, 4315, 4316, 4317, 4318, 4378, 4379, 4380, 4381, 4382, 4412,
        4413, 4420, 4421, 4434, 4452, 4453]

            option = st.selectbox(
                'Select a financial statement', my_list)

            df_new = pd.read_csv('resources/data/df_new.csv')

            # Convert date column to datetime
            # df1['FinancialsDate'] = pd.to_datetime(df1['FinancialsDate', 'Year', 'Month', 'Week', 'Day', 'Date','ReturnEquityRatio'])
            df1 = df_new.drop(
                ['FinancialsDate', 'Year', 'Month', 'Week', 'Day', 'Date', 'ReturnEquityRatio'], axis=1)

            # Encode categorical columns using one-hot encoding
            encoder = OneHotEncoder()
            encoded_cat_columns = encoder.fit_transform(df1[['Financial_Type', 'Country', 'Industry']])
            encoded_cat_columns_df = pd.DataFrame(encoded_cat_columns.toarray(),
                                                  columns=encoder.get_feature_names(
                                                      ['Financial_Type', 'Country', 'Industry']))

            # Combine encoded categorical columns with numerical columns
            X = pd.concat(
                [df1.drop(['Financial_Type', 'Country', 'Industry'], axis=1), encoded_cat_columns_df],
                axis=1)

            iforest = IsolationForest(max_samples='auto', bootstrap=False, contamination=0.03, n_jobs=-1, random_state=42)
            iforest_ = iforest.fit(X)
            y_pred = iforest_.predict(X)

            y_score = iforest.decision_function(X)
            neg_value_indices = np.where(y_score < 0)

            exp = shap.TreeExplainer(iforest)  # Explainer
            shap_values = exp.shap_values(X)  # Calculate SHAP values
            shap.initjs()

            st.markdown('**Force plot for selected financial statement**')

            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(exp.expected_value, shap_values[option, :], X.iloc[option, :]))
            st.markdown("A force plot is a visual that shows the influence of feature(s) on the predictions.")


            st.markdown('**Bar plot for selected financial statement**')
            shap.initjs()
            explainer = shap.Explainer(iforest, X)
            shap_values = explainer(X)
            fig = shap.plots.bar(shap_values[option])
            st.pyplot(fig)

            #fig2 = shap.plots.waterfall(shap_values[option])
            #st.pyplot(fig2)


    if page_selection == "Step 1: Input":
        df = pd.read_csv('resources/data/financials_data.csv')
        st.title('Input')
        st.markdown("In this section, the user can view the data that was used as an input to the model. Input data are are he financial statements that will be used for analysis. The financial statements include, balance sheets, income statements, and cash flow statements, "
                    "in a specific format. The app validates the data entered by the user and provide feedback in case of any errors or inconsistencies")

        st.warning('The graphs that appear are interactive and can be zoomed in or certain data can be selected')


        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            df2 = uploaded_file.read()
            st.write("filename:", uploaded_file.name)


        if st.checkbox("Show raw data"):
            st.subheader("Financial Statements")
            st.dataframe(df)  # Same as st.write(df)
            st.subheader("Summary statistics of the dataset")
            df1 =df.describe()
            st.table(df1)


        if st.checkbox("Show explanation"):
            st.subheader("The input dataset")
            st.markdown("The data consists of 1,000s of financial statements (i.e. income statement, balance sheet, cash flow statement, financial type, year), details of the company (industry, age) and "
                        "whether the company defaulted in the 12-months post the date of the financial statement.")

            st.markdown("- There are ~8,000 rows (i.e. financial statements) and ~100 columns (financial entries, company details and default indicator")
            st.markdown("- There is a time component, which is one dimension we want to analyse the predicted fraud by")

            st.subheader("Exploratory Data Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.image('resources/imgs/financial_type.png',use_column_width=True, caption= 'The plot shows the frequencies of the different financial types')

            with col2:
                st.image('resources/imgs/industry.png',use_column_width=True, caption= 'The plot shows the frequencies of the different industry types')


            st.markdown('The most frequent financial types are Audited - Signed (3341) and Financials - By Accounting Officer - Signed (916). These two make up more than 90% of dataset')
            st.markdown("Checking through the Industry type, the most frequent industry types are Construction (2030) and Unknown (1759). The industry named 0 will be added to the UNKNOWN industry as it is also an unknown industry")

            st.subheader("Region of financial statements")
            st.image('resources/imgs/country.png',use_column_width=True, caption='The most frequent countries are South Africa (2663) and UNKNOWN (1762) which is more than 90%. The Country named AFRICA will be captured in UNKNOWN country as there is no country named AFRICA')

            st.subheader("Dates of financial statements")
            st.image('resources/imgs/dates_percentage.png', caption='From the plot, it can be deduced that the financial statements in not evenly distributed across the years, There are 11% (highest value) financial year end in 2017 February followed by 9% in 2018, February. Others that presented in January, March - November makes up 38.8% of the distribution.')

            st.subheader("Plot the proportion of default in dataset")
            st.image('resources/imgs/default.png')
            st.markdown("Only 4.4% (203) transactions in the dataset are default while 95.6% (4382) transactions are nondefault with ratio of 21.59 indicating hight class imbalance in the dataset. Building a machine learning model on a highly skewed data as shown here, the nondefault transactions will influence the training of the model almost entirely, thus affecting the results.")

if __name__ == '__main__':
    main()
