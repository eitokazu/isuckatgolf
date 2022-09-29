import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import numpy as np
import pyodbc as db
import sqlalchemy as sa
import pickle

# # DB connect

# server = 'LAPTOP-5F2B3G3Q\SQLEXPRESS' 
# database = 'golf_tracker' 
# conn = db.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database)
# cursor = conn.cursor()

# # Get data
# golf = pd.read_sql("SELECT * FROM hole_data", conn)

# # Format data
# golf['Approach1'] = golf['Approach1'].astype(float)
# golf['Scramble'] = 0
# golf.loc[(golf['Shots']==golf['Par']) & (golf['Putts']<=1), 'Scramble'] = 1
# golf['Par_Diff'] = golf['Shots'] - golf['Par']



golf = pd.read_pickle('golf.pkl')


# ----------------------


# Set up data for use in par_diff linear regression
cols = ['Length', 'Par', 'Approach1', 'Fairway', 'GIR']
X = golf[cols]
Y = golf['Par_Diff']


# Sidebar

st.sidebar.header('Specify Input Parameters')
# Set up sidebar and input parameters
def user_input_features():
    Length = st.sidebar.slider('Length', float(X.Length.min()), float(X.Length.max()), float(X.Length.mean()))
    Par = st.sidebar.radio('Par', [3,4,5], index=1)
    Approach1 = st.sidebar.slider('Approach1', float(X.Approach1.min()), float(X.Approach1.max()), float(X.Approach1.mean()))
    Fairway = st.sidebar.checkbox('Fairway')
    GIR = st.sidebar.checkbox('GIR')
    # Putts = st.sidebar.slider('Putts', float(X.Putts.min()), float(X.Putts.max()), float(X.Putts.mean()))
    base_data = {'Length': Length,
            'Par': Par,
            'Approach1': Approach1,
            'Fairway': Fairway,
            'GIR': GIR,
            # 'Putts': X.Putts.mean(),
            }
    features = pd.DataFrame(base_data, index=[0])
    return features

df = user_input_features()

# Build Regression Model

# Sike, read in pickled model
model = pickle.load(open('model.pkl', 'rb'))

# Apply Model to Make Prediction
prediction = model.predict(df)

# Anotha one, load pickled standardized model
model_s = pickle.load(open('model_s.pkl', 'rb'))
model_s.score(X,Y)
coefs_s = pd.DataFrame(model_s[1].coef_,
    columns=['Coefficients'],
    index=X.columns
)
coefs_s = coefs_s.abs()

# Build Radar plot of features for importance ranking
# Radar function
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

# Radar plot layout
N = len(cols)
theta = radar_factory(N, frame='polygon')
fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                        subplot_kw=dict(projection='radar'))
fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
ax.grid(True)
ax.set_title('Feature Importance for Par Differential', weight='bold', size='medium', position=(0.5, 1.1),
                horizontalalignment='center', verticalalignment='center')
    # for d, color in zip(case_data, colors):
ax.plot(theta, coefs_s, color='r')
ax.fill(theta, coefs_s, facecolor='r', alpha=0.25, label='_nolegend_')
ax.set_varlabels(cols)

def approach_bin_setup():
    # define bins
    bins = [0, 50, 75, 100, 120, 140, 160, 180, 200, 230, 999]
    bin_labs = ['0-50', '51-75', '76-100', '101-120', '121-140', '141-160', '161-180', '181-200', '201-230', '230+']
    golf_tmp = golf.copy()
    golf_tmp['Approach_Bin'] = pd.cut(x=golf_tmp['Approach1'], bins=bins, labels=bin_labs, include_lowest=True)
    # Break into GIR and FW vars from non-par5 holes
    app_binned = golf_tmp.loc[golf['Par'] != 5]
    app_binned_gir = app_binned.loc[app_binned['GIR']=='1']
    app_binned_gir_fw = app_binned_gir.loc[app_binned_gir['Fairway']=='1']
    # Get totals in each bin
    bin_tots = app_binned.Approach_Bin.value_counts()
    bin_gir_tots = app_binned_gir.Approach_Bin.value_counts()
    bin_gir_fw_tots = app_binned_gir_fw.Approach_Bin.value_counts()
    # Format totals to %
    bin_pct = (bin_gir_tots / bin_tots) * 100
    bin_fw_pct = (bin_gir_fw_tots / bin_tots) * 100
    bin_nonfw = bin_pct - bin_fw_pct
    # Get bin category value as lists
    bin_pct_labs = bin_pct.index.categories.to_list()
    bin_pct_vals = bin_pct.values
    # Set up bar chart
    fig2, ax2 = plt.subplots()
    ax2.bar(bin_pct_labs, bin_fw_pct.values, width=0.4, label='Fairway', color='g')
    ax2.bar(bin_pct_labs, bin_nonfw.values, width=0.4, bottom=bin_fw_pct.values, label='Non-Fairway', color='b')

    ax2.set_ylabel('Green Pct')
    ax2.set_xlabel('Yardage Bin')
    ax2.set_title('% of Greens Hit from Range')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    return fig2

def round_stats():
    data = golf.copy()
    data[['Fairway', 'GIR', 'Hazard']] = data[['Fairway', 'GIR', 'Hazard']].apply(pd.to_numeric)
    round_stats = data.groupby(['Date', 'Course'])['Par', 'Shots', 'Par_Diff', 'Fairway', 'GIR', 'Putts', 'Hazard', 'Scramble'].sum()
    return round_stats

def scorecard_stats():
    data = golf.copy()
    data[['Fairway', 'GIR', 'Hazard']] = data[['Fairway', 'GIR', 'Hazard']].apply(pd.to_numeric)
    fw = data['Fairway'].sum()
    gir = data['GIR'].sum()
    p = data['Putts'].mean()
    prox = data['Hole_Prox'].mean()
    return [fw, gir, p, prox]

'''BEGIN STREAMLIT APP'''


# Title
st.write("""
# Personal golf stats for Eric
""")
st.write('---')

# Main Panel

# Print specified input parameters
st.header('Score Prediction Based on Specified Input Parameters')
col1, col2 = st.columns(2)
col1.write(df)
col2.metric(label='Predicted Score', value=round(prediction[0], 4))
st.write('---')

# Feature importance radar plot display
st.header('Feature Importance')
plt.title('Feature importance based on scaled regression')
st.pyplot(fig)
st.write('---')

# Basic stats from the card
st.header('Basic hole stats')
bs1, bs2, bs3, bs4 = st.columns(4)
bs = scorecard_stats()
bs1.metric(label=r'% Fairways hit', value=(round((bs[0]/len(golf))*100, 3)))
bs2.metric(label=r'% Greens in Reg', value=round((bs[1]/len(golf))*100, 3))
bs3.metric(label='Avg Putts', value=round(bs[2], 3))
bs4.metric(label='Avg 1st Putt Dist', value=round(bs[3], 3))
st.write('---')

# Stats by round
st.header('Full round numbers')
round_table = round_stats()
st.write(round_table)
st.write('---')

# Approach performance binned bar chart
st.header('Approach performance by yardage')
test = approach_bin_setup()
st.write(test)
st.write('---')

