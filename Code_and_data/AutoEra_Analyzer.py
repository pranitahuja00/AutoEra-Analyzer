import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

st.title("AutoEra Analyzer")

tab1, tab2, tab3, tab4 = st.tabs(['Data Overview', 'Data Analysis', 'Trends over years', 'Interactive Analysis and Plotting'])

cars = pd.read_csv('https://raw.githubusercontent.com/pranitahuja00/CMSE830/main/midSem_project/Car%20Dataset%201945-2020.csv', delimiter=",", skiprows=0)


# TAB 1
with tab1:
    st.subheader("About the data:")
    st.write('This dataset contains a comprehensive list of cars manufactured from 1935 to 2020. It includes details such as make, model, year, engine size, fuel type, transmission type, drivetrain, body style, number of doors, and many more specifications. The purpose of this dataset is to provide a comprehensive list of car specifications that can be used for various research and analysis purposes, such as market and trend analysis.')
    st.write('Raw Data rows/ records: ', cars.shape[0])
    st.write('Raw Data columns/ features: ', cars.shape[1])

#Dropping columns having more than 50% missing values
dropped_cols = {'Columns':[], 'Missing_perc':[]}
for i in cars.columns:
    if(cars[i].isna().sum()/cars.shape[0]>0.5):
        if(i not in ['boost_type', 'presence_of_intercooler']):
            dropped_cols['Columns'].append(i)
            dropped_cols['Missing_perc'].append(cars[i].isna().sum()/cars.shape[0])
            cars.drop(i, axis=1, inplace=True)
dropped_cols = pd.DataFrame(dropped_cols)

with tab1:
    st.write("First, I checked for missing data and dropped all features which were missing values for more than 50 percent of the total records and didn't have much use for us.")
    st.write('The following columns were dropped for missing majority of the data: -',dropped_cols)

#Dropping some more useless columns
cars.drop(['id_trim', 'year_to', 'number_of_seats', 'minimum_trunk_capacity_l', 'full_weight_kg', 'turnover_of_maximum_torque_rpm', 'engine_hp_rpm', 'back_suspension', 'rear_brakes', 'city_fuel_per_100km_l', 'highway_fuel_per_100km_l', 'fuel_grade'], axis=1, inplace=True)
#Renaming
cars.rename(columns={'Modle':'Model', 'length_mm':'length', 'width_mm':'width', 'height_mm': 'height', 'wheelbase_mm':'wheelbase', 'front_track_mm':'front_track', 'rear_track_mm':'rear_track', 'curb_weight_kg':'weight', 'ground_clearance_mm':'ground_clearance', 'max_trunk_capacity_l':'trunk_capacity', 'maximum_torque_n_m':'torque', 'number_of_cylinders':'cylinders', 'engine_type':'fuel', 'presence_of_intercooler':'intercooler', 'capacity_cm3':'displacement', 'engine_hp':'horsepower', 'turning_circle_m':'turning_radius', 'mixed_fuel_consumption_per_100_km_l':'avg_kmpl', 'fuel_tank_capacity_l':'fuel_capacity', 'acceleration_0_100_km/h_s':'acceleration', 'max_speed_km_per_h':'top_speed', 'front_brakes':'brakes', 'front_suspension':'suspension', 'number_of_gears':'gears', 'year_from':'year'}, inplace=True)
# Feature Engineering
cars['bs_ratio'] = cars['cylinder_bore_mm']/cars['stroke_cycle_mm']
cars.drop(['cylinder_bore_mm', 'stroke_cycle_mm'], axis=1, inplace=True)

#Cleaning
cars['boost_type'].replace('none', 'Naturally Asp', inplace=True)
cars['boost_type'].replace('Intercooler', 'Turbo', inplace=True)
cars['boost_type'].fillna('Naturally Asp', inplace=True)
cars['intercooler'].fillna('No', inplace=True)
cars['intercooler'].replace('Present', 'Yes', inplace=True)
cars['fuel']=cars['fuel'].str.upper()
for i in ['GASOLINE', 'GASOLINE, GAS', 'GAS']:
    cars['fuel'].replace(i, 'PETROL', inplace=True)
for i in ['GASOLINE, ELECTRIC', 'DIESEL, HYBRID']:
    cars['fuel'].replace(i, 'HYBRID', inplace=True)
cars['fuel'].replace('LIQUEFIED COAL HYDROGEN GASES', 'HYDROGEN', inplace=True)
for i in ['Multi-point fuel injection', 'Injector','direct injection', 'Monoinjection', 'Common Rail','distributed injection (multipoint)', 'direct injection (direct)','Central injection (single-point or single-point)','combined injection (direct-distributed)', 'Central injection','the engine is not separated by the combustion chamber (direct fuel injection)']:
    cars['injection_type'].replace(i, 'Fuel Injector', inplace=True) 
cars['body_type'].replace('Hatchback 3 doors', 'Hatchback', inplace=True)
cars['cylinder_layout']=cars['cylinder_layout'].str.upper()
cars['cylinder_layout'].replace('-', np.nan, inplace=True)
cars['cylinder_layout'].replace('V-TYPE WITH SMALL ANGLE', 'V-TYPE', inplace=True)
cars['cylinder_layout'].replace('ROTARY-PISTON', 'ROTOR', inplace=True)
cars['cylinder_layout'].replace('ROTARY', 'ROTOR', inplace=True)
cars['drive_wheels'].replace('Rear wheel drive', 'RWD', inplace=True)
cars['drive_wheels'].replace('Front wheel drive', 'FWD', inplace=True)
cars['drive_wheels'].replace('All wheel drive (AWD)', 'AWD', inplace=True)
cars['drive_wheels'].replace('Four wheel drive (4WD)', '4WD', inplace=True)
cars['drive_wheels'].replace('full', '4WD', inplace=True)
cars['drive_wheels'].replace('Constant all wheel drive', '4WD', inplace=True)
for i in ['robot','Continuously variable transmission (CVT)','Electronic with 1 clutch', 'Electronic with 2 clutch']:
    cars['transmission'].replace(i, 'Automatic', inplace=True)
for i in ['ventilated disc','Disc', 'Disc ventilated','Disc composite, ventilated', 'Disc composite','ventilated ceramic', 'ventilated disc, perforated','Disk ceramic']:
    cars['brakes'].replace(i,'disc', inplace=True)
cars['brakes'].replace('N/a', np.nan, inplace=True)
cars.drop(cars.columns[0], axis=1,inplace=True)

with tab1:
    st.write("After removing some columns, I created a few features such as 'bs_ratio'. Bore-Stroke ratio ('bs_ratio') is the ratio of the piston diameter and the stroke length. The dataset had two columns for the piston diameter and stroke length, so I calculated the ratio as that is more useful and dropped the bore and stroke columns.")
    st.write("A preview of the new dataset: -", cars.head())
    st.write("New shape: ", cars.shape)

    st.write("Categorzing the attributes: -")
categorical_attr = ['Make', 'Model', 'Generation', 'Series', 'Trim', 'body_type', 'injection_type', 'cylinder_layout', 'fuel', 'boost_type', 'intercooler', 'drive_wheels', 'transmission', 'brakes', 'suspension']
continuous_attr = ['length', 'width', 'height', 'wheelbase', 'front_track', 'rear_track','weight', 'ground_clearance', 'trunk_capacity', 'torque','displacement','horsepower','turning_radius', 'avg_kmpl', 'fuel_capacity', 'acceleration', 'top_speed','bs_ratio']
discrete_attr = ['year', 'cylinders', 'valves_per_cylinder', 'gears']

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Categorical: -',categorical_attr)

    with col2:
        st.write('Continuous: -',continuous_attr)

    with col3:
        st.write('Discrete: -',discrete_attr)



# TAB 2
with tab2:
    st.subheader("Visualizing and studying data trends and relationships:")

counts = cars['year'].value_counts().reset_index()
counts.columns = ['year', 'Count']
counts['year']=counts['year'].astype(int)

# Sort the DataFrame by 'Value' if needed
counts = counts.sort_values(by='year')

with tab3:
    st.write('Distribution of the car records over the years:')
chart=alt.Chart(counts).mark_line().encode(
    x='year',
    y='Count'
).interactive()



# TAB 3
with tab3:
    st.altair_chart(chart, use_container_width=True)
    st.write("We can see an upward trend along the years till 2008 which was the peak followed by a decline.")

    st.write("Let's check some trends over the years: ")
    st.write("Horsepower: ")
cars.drop(cars[cars['horsepower']>1000].index, axis=0, inplace=True)
hp_year_chart = alt.Chart(cars).mark_circle().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y='horsepower',
).interactive()
hp_year_chart_line = hp_year_chart.transform_regression('year', 'horsepower').mark_line()

with tab3:
    st.altair_chart(hp_year_chart+hp_year_chart_line, use_container_width=True)
    st.write("The fit line shows a slight positive trend that's because of the dip in horsepower figures after the 1970s started.")

    st.write("Acceleration: ")
acc_year_chart = alt.Chart(cars).mark_circle().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y='acceleration',
).interactive()
acc_year_chart_line = acc_year_chart.transform_regression('year', 'acceleration').mark_line()

with tab3:
    st.altair_chart(acc_year_chart+acc_year_chart_line, use_container_width=True)
    st.write("We can see that with time average time to accelerate from 0-100 kmph has gone down due to advances in engineering.")

    st.write("Fuel Economy:")
kmpl_year_chart = alt.Chart(cars).mark_circle().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y='avg_kmpl',
).interactive()
kmpl_year_chart_line = kmpl_year_chart.transform_regression('year', 'avg_kmpl').mark_line()

with tab3:
    st.altair_chart(kmpl_year_chart+kmpl_year_chart_line, use_container_width=True)
    st.write("According to this dataset the fuel economy of cars has gotten worse over time and is on the decline.")

cars['displacement']=cars['displacement'].astype(float)
disp_year_chart = alt.Chart(cars).mark_circle().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y='displacement',
).interactive()
disp_year_chart_line = disp_year_chart.transform_regression('year', 'displacement').mark_line()

with tab3:
    st.altair_chart(disp_year_chart+disp_year_chart_line, use_container_width=True)
    st.write("Average displacement of cars has gone down along the years as we move forward to more fuel efficient vehicles which can give better performance with a smaller engine.")



# TAB 4
with tab4:
    st.subheader("Check the relationship between attributes:")
    selected_column1 = st.selectbox("Select attribute", continuous_attr, key=2)
    selected_column2 = st.selectbox("Select attribute", continuous_attr, key=3)
    chart_line_check = st.checkbox("Show Regression Line", key=1)
    if selected_column1:
        chart2 = alt.Chart(cars).mark_circle().encode(x=selected_column1, y=selected_column2).interactive()
        chart_line2 = chart2.transform_regression(selected_column1, selected_column2).mark_line()
        st.altair_chart(chart2+chart_line2 if chart_line_check else chart2, theme="streamlit", use_container_width=True)