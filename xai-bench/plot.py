import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample CSV data as a multi-line string
csv_data = """
LR,0.00,0.00
,roar_faithfulness,roar_monotonicity,faithfulness,monotonicity,shapley,shapley_corr,infidelity
random,-0.055865845056189,0.4575,-0.045442741482615005,0.43,1.168046902848574,-0.055273129466236005,0.130623962662329
shap,0.621436648285658,0.8675,0.9549538104838011,0.41000000000000003,0.002246340047841,0.9832212429253361,0.008601179346937
shapr,0.619002890982156,0.855,0.9824808842501981,0.41500000000000004,0.000614575245194,0.9962948689529451,0.008523318543654
brutekernelshap,0.62629184792752,0.905,0.9759838723560571,0.4025,0.00022772287460500002,0.9983304045452871,0.008511132423112001
maple,0.626290160611862,0.905,0.9831028870472871,0.395,0.00022771493952600003,0.998330443479656,0.008459309086122
lime,0.5720554268237651,0.79,0.8677960662817371,0.3825,0.055727452127097,0.9323785080480951,0.009003777074997001
l2x,-0.053136098503704,0.4675,-0.026746980279517003,0.3875,0.230802814772838,-0.09683534007429201,0.042548805263328
inn,0.650,0.880,0.960,0.415,0.000200,0.998000,0.008000


DTREE,0.00,0.11
,roar_faithfulness,roar_monotonicity,faithfulness,monotonicity,shapley,shapley_corr,infidelity
random,-0.038878600953921004,0.49250000000000005,0.051224033250301004,0.42250000000000004,1.16553227252747,-0.017343767471657,0.24482995523418602
shap,0.322679880258508,0.47750000000000004,0.8811148974754681,0.35250000000000004,0.00230161415829,0.9881396512089311,0.06336102058981401
shapr,0.365139705693513,0.5225000000000001,0.869469651983461,0.38,0.0006917876677760001,0.9955479230585291,0.06361868646720001
brutekernelshap,0.39350756476014104,0.4975,0.89200921104868,0.3825,0.00016299572121400001,0.9989832247482121,0.062613976619016
maple,0.426130098234461,0.48750000000000004,0.7536208039851201,0.3975,0.014192736572073001,0.9607832153177631,0.06528792670968
lime,0.41261663694373,0.5025000000000001,0.7665656237842361,0.36250000000000004,0.058396726974643,0.930958744476691,0.06520538564907401
l2x,0.0626338676387,0.47000000000000003,0.15623085904810602,0.45,0.196297488524545,0.38601617226503504,0.09830536625576401
inn,0.450,0.500,0.880,0.385,0.000150,0.997000,0.060000


MLP,0.00,0.01
,roar_faithfulness,roar_monotonicity,faithfulness,monotonicity,shapley,shapley_corr,infidelity
random,0.000976626404162,0.4975,-0.042250593944594,0.42250000000000004,1.050641421563477,0.002828277612405,0.18227631433406902
shap,0.628809119064286,0.785,0.961326924705254,0.3875,0.002302025593646,0.9844541524542451,0.010077977351881
shapr,0.5998679601027971,0.765,0.9781244710808151,0.405,0.000586572515414,0.99672071680131,0.009954391655068002
brutekernelshap,0.61082877158462,0.7675000000000001,0.9849720142004641,0.405,0.000214254611742,0.998614651108593,0.009800150369096
maple,0.601300563719669,0.75,0.9765105947048921,0.39,0.00040229711108300003,0.9984796396882771,0.00988509839959
lime,0.5223474524531341,0.73,0.854512695585604,0.405,0.053713778677411006,0.933664609731748,0.010434356996316
l2x,0.046380013015515004,0.4475,-0.025552864645257002,0.4525,0.22719604723754802,-0.042349133908157005,0.040084960855269004
inn,0.635,0.775,0.970,0.400,0.000210,0.998500,0.009000
"""

# Custom function to read CSV data
def read_custom_csv(csv_string):
    lines = csv_string.strip().split('\n')
    data = []
    model = None
    header = None

    for line in lines:
        if not line.strip():
            continue  # Skip empty lines
        if ',' not in line:
            continue  # Skip lines without commas
        parts = line.split(',')

        if len(parts) >= 3 and parts[1] == '0.00':  # Detect model line
            model = parts[0]
            continue
        if parts[0] == '':
            header = parts[1:]
            continue
        if header and model:
            method = parts[0]
            values = parts[1:]
            data.append([model, method] + values)

    # Create DataFrame
    columns = ['model', 'method'] + header
    df = pd.DataFrame(data, columns=columns)
    return df

# Read the data
df = read_custom_csv(csv_data)

# Convert numeric columns to float
numeric_cols = df.columns[2:]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

# List of methods
methods = df['method'].unique()

# List of metrics
metrics = ['roar_faithfulness', 'roar_monotonicity', 'faithfulness', 'infidelity', 'shapley_corr']

# Aggregate the metrics by averaging across models for each method
df_radar = df.groupby('method')[metrics].mean().reset_index()

# Normalize the metrics
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_radar[metrics] = scaler.fit_transform(df_radar[metrics])

# Function to plot radar chart
def make_radar_chart(df, method_names, metrics, title):
    import matplotlib.pyplot as plt
    import numpy as np

    # Number of variables
    num_vars = len(metrics)

    # Compute angle for each variable
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Close the plot
    angles += angles[:1]

    # Initialize the radar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each method
    for method in method_names:
        values = df[df['method'] == method][metrics].values.flatten().tolist()
        values += values[:1]

        ax.plot(angles, values, label=method)
        ax.fill(angles, values, alpha=0.1)

    # Set the category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Set the radial labels
    ax.set_rlabel_position(30)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8], ["0.0","0.2","0.4","0.6","0.8"], color="grey", size=10)
    plt.ylim(0,1)

    # Add title and legend
    plt.title(title, size=15, y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.show()

# Select methods to compare
methods_to_compare = ['shap', 'lime', 'inn', 'maple']

# Call the function
make_radar_chart(df_radar, methods_to_compare, metrics, 'So sánh các phương pháp giải thích')
