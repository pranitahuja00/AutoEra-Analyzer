import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from altair import datum

st.title("AutoEra Analyzer")

tab1, tab2, tab3, tab4 = st.tabs(['Project Overview', 'Data Analysis', 'Trends over years', 'Conclusion'])

cars = pd.read_csv('https://raw.githubusercontent.com/pranitahuja00/CMSE830/main/midSem_project/Car%20Dataset%201945-2020.csv', delimiter=",", skiprows=0)


# TAB 1
with tab1:
    st.subheader("Objective:")
    st.write("This data science dashboard aims to provide valuable insights and trends in the world of automobiles over the past several decades. This dashboard will enable users to explore, visualize, and extract meaningful information from a comprehensive car dataset, helping enthusiasts, researchers, and industry professionals gain a deeper understanding of automotive evolution, performance, and technological advancements.")
    st.subheader("About the data:")
    st.write("Data Source: [Kaggle - Car Specification Dataset 1945-2020](https://www.kaggle.com/datasets/jahaidulislam/car-specification-dataset-1945-2020/data)")
    st.write('This dataset contains a comprehensive list of cars manufactured from 1945 to 2020. It includes details such as make, model, year, engine size, fuel type, transmission type, drivetrain, body style, number of doors, and many more specifications. The purpose of this dataset is to provide a comprehensive list of car specifications that can be used for various research and analysis purposes, such as market and trend analysis.')
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
for i in ['robot','continuously variable transmission (cvt)','electronic with 1 clutch', 'electronic with 2 clutch']:
    cars['transmission'].replace(i, 'automatic', inplace=True)
for i in ['ventilated disc','Disc', 'Disc ventilated','Disc composite, ventilated', 'Disc composite','ventilated ceramic', 'ventilated disc, perforated','Disk ceramic']:
    cars['brakes'].replace(i,'disc', inplace=True)
cars['brakes'].replace('N/a', np.nan, inplace=True)
cars.drop(cars.columns[0], axis=1,inplace=True)

with tab1:
    st.write("After dropping these columns, I had to perform manual data cleaning by going through each categorical column and renaming certain value to maintan a proper format and avoif having same values under different names which would negatively affect my visualizations.")
    st.write("A preview of the new dataset: -", cars.head())
    st.write("New shape: ", cars.shape)

    st.write("Categorzing the attributes: -")
categorical_attr = ['make', 'model', 'generation', 'series', 'trim', 'body_type', 'injection_type', 'cylinder_layout', 'fuel', 'boost_type', 'intercooler', 'drive_wheels', 'transmission', 'brakes', 'suspension']
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

    st.write("Since the dataset contains more than 70,000 records, I have taken random samples from the data to create a smaller dataframe for the data analysis part which will preserve the data trends and relationships while giving clearer and lighter visualizations which will take less loading time.")
    

def countFunc(column_name, data_type):
    counts = cars[column_name].value_counts().reset_index()
    counts.columns = [column_name, 'count']
    counts[column_name]=counts[column_name].astype(data_type)
    counts = pd.DataFrame(counts)
    counts['year_1'] = cars['year']
    return counts

# TAB 2
with tab2:
    st.subheader("Visualizing and studying data trends and relationships:")
    
make_dist_chart=alt.Chart(countFunc('make',str)).mark_bar().encode(
    x='make',
    y=alt.Y('count').scale(type="log")
).interactive().properties(width=800, height=500)

with tab2:
    st.write("Car manufacturer distribution:")
    st.altair_chart(make_dist_chart)
    st.write("We can see that most of the cars in the dataset have been manifactured by Cheverolet, followed by other top manufacturers such as Toyota, Volkswagen, Mercedez-Benz, BMW, Nissan, Audi, Opel and Mazda.")

trans_dist_chart=alt.Chart(countFunc('transmission',str)).mark_bar().encode(
    x='transmission',
    y=alt.Y('count').scale(type="log"),
    color='transmission'
).interactive().properties(width=800, height=400)


with tab2:
    st.write("Car transmission distribution:")
    st.altair_chart(trans_dist_chart)
    st.write("We can see that manual transmission cars are dominating the dataset by a small margin here, an in depth analysis over time can be seen on the next tab.")

fuel_dist_chart=alt.Chart(countFunc('fuel',str)).mark_bar().encode(
    x='fuel',
    y=alt.Y('count').scale(type="log"),
    color='fuel'
).interactive().properties(width=800, height=400)

with tab2:
    st.write("Car fuel type distribution:")
    st.altair_chart(fuel_dist_chart)
    st.write("It is very clear that Petrol engines have been the most widely used engines for cars followed by Diesel then Hybrid, Electic motors, Hydrogen ones and Rotors.")

cyl_dist_chart=alt.Chart(countFunc('cylinders',str)).mark_bar().encode(
    x='cylinders',
    y=alt.Y('count').scale(type="log"),
    color='cylinders'
).interactive().properties(width=800, height=400)

with tab2:
    st.write("Car cylinders distribution:")
    st.altair_chart(cyl_dist_chart)
    st.write("We can clearly see that 4 cylinder cars have been the most common road cars followed by 6 and 8 cylinders ones. It is very rare for cars to have 1 or 7 cylinders, and only a few expensive hypercars can be seen with 16 cylinders.")

cylLayout_dist_chart=alt.Chart(countFunc('cylinder_layout',str)).mark_bar().encode(
    x='cylinder_layout',
    y=alt.Y('count').scale(type="log"),
    color='cylinder_layout'
).interactive().properties(width=800, height=400)

cars_sample = cars.sample(n = round(cars.shape[0]/15))

with tab2:
    st.write("Car cylinder layouts distribution:")
    st.altair_chart(cylLayout_dist_chart)
    st.write("The chart shows that an Inline cylinder layout is the most widely used among road cars, followed by V-Type layout which is mostly seen in high performance supercars. Opposed layout comes in at third followed by W-Type which is only seen in really expensive hypercars and then Rotor which was just used in some very old cars.")

    st.subheader("Check the relationship between attributes:")
    selected_column1 = st.selectbox("Select attribute", continuous_attr, key=2)
    selected_column2 = st.selectbox("Select attribute", continuous_attr, key=3)
    chart_line_check = st.checkbox("Show Regression Line", key=1, value=True)
    if selected_column1:
        chart2 = alt.Chart(cars_sample).mark_circle().encode(x=selected_column1, y=selected_column2).interactive()
        chart_line2 = chart2.transform_regression(selected_column1, selected_column2).mark_line(color='red')
        st.altair_chart(chart2+chart_line2 if chart_line_check else chart2, theme="streamlit", use_container_width=True)



# TAB 3
with tab3:
    st.subheader("Visualizing and understanding how attributes were affected over the years:")
    st.write('Distribution of the car records over the years:')
chart=alt.Chart(countFunc('year',int)).mark_line().encode(
    x='year',
    y='count'
).interactive()

with tab3:
    st.altair_chart(chart, use_container_width=True)
    st.write("We can see an upward trend along the years till the early 2000s which was a peak era followed by a decline.")


trans_line_chart=alt.Chart(cars).mark_line().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y=alt.Y('count()').scale(type="log"),
    color='transmission'
).transform_filter(datum.transmission!=None).interactive().properties(width=800, height=400)

with tab3:
    st.write("Transmission types over time:")
    st.altair_chart(trans_line_chart)
    st.write("Both manual and automatic transmissions have been equally popular in cars. Manual transmission being the older one takes the majority among older cars but automatic started to get more popular duting the early 2000s.")

fuel_line_chart=alt.Chart(cars).mark_line().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y=alt.Y('count()').scale(type="log"),
    color='fuel'
).transform_filter(datum.fuel!=None).interactive().properties(width=800, height=400)

with tab3:
    st.write("Fuel types over time:")
    st.altair_chart(fuel_line_chart)
    st.write("Petrol can be seen as the oldest fuel types for cars in this chart which obviously has been the most popular fuel type followed bu Diesel which was introduced later but couldn't catch up to Petrol. Both these fuel types however started to decline in popularity after the early 2000s as people started becoming more conscious abou the environment and started moving towards Hybrid, electric and even Hydrogen based vehicles.")

cyl_line_chart=alt.Chart(cars).mark_circle().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y=alt.Y('count()').scale(type="log"),
    color='cylinders:N'
).transform_filter(datum.cylinders!=None).interactive().properties(width=800, height=400)

with tab3:
    st.write("Number of cylinders over time:")
    st.altair_chart(cyl_line_chart)
    st.write("4 cylinders engines have always been one of the most popular ones for several decades and became the most popular during the 1970s era before which 8 cylinder engines held the throne. Now 4 cylinder ones are followed by 6 cylidner engines used in performance sports cars. 10 and 12 cylinder engines also started to become popular in high performance supercars starting from 1980s.")

cylLayout_line_chart=alt.Chart(cars).mark_line().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y=alt.Y('count()'),
    color='cylinder_layout:N'
).transform_filter(datum.cylinder_layout!=None).interactive().properties(width=800, height=400)

with tab3:
    st.write("Cylinder layouts over time:")
    st.altair_chart(cylLayout_line_chart)
    st.write("Inline cylinder layoust were the first ones to be used in road cars as conveyed by this chart which was followed by the Opposed layout which didn't take off much, but V-Type engines started getting popular during the 1950s and became the second most popular engine layouts and the most popular among expensive performance cars. However, Inline layouts are still the most common among Inernal Combustion Engine cars as they are present in almost every normal road car.")

cars_sample.drop(cars_sample[cars_sample['horsepower']>1000].index, axis=0, inplace=True)
hp_year_chart = alt.Chart(cars_sample).mark_circle().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y='horsepower'
).interactive()
hp_year_chart_line = hp_year_chart.transform_regression('year', 'horsepower').mark_line(color='red')

with tab3:
    st.write("Horsepower: ")
    st.altair_chart((hp_year_chart+hp_year_chart_line), use_container_width=True)
    st.write("The fit line shows a slight positive trend instead of a strong one that's because of the dip in horsepower figures after the 1970s started.")

    
acc_year_chart = alt.Chart(cars_sample).mark_circle().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y='acceleration',
).interactive()
acc_year_chart_line = acc_year_chart.transform_regression('year', 'acceleration').mark_line(color='red')

with tab3:
    st.write("Acceleration: ")
    st.altair_chart(acc_year_chart+acc_year_chart_line, use_container_width=True)
    st.write("We can see that with time average time to accelerate from 0-100 kmph has gone down due to advances in engineering.")

    
speed_year_chart = alt.Chart(cars_sample).mark_circle().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y='top_speed',
).interactive()
speed_year_chart_line = speed_year_chart.transform_regression('year', 'top_speed').mark_line(color='red')

with tab3:
    st.write("Top Speed:")
    st.altair_chart(speed_year_chart+speed_year_chart_line, use_container_width=True)
    st.write("The top speed (km/h) of cars has increased over the decades and is still on the rise due to advances in engineering and aerodynamics.")

cars_sample['displacement']=cars_sample['displacement'].astype(float)
disp_year_chart = alt.Chart(cars_sample).mark_circle().encode(
    x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])),
    y='displacement',
).interactive()
disp_year_chart_line = disp_year_chart.transform_regression('year', 'displacement').mark_line(color='red')

with tab3:
    st.altair_chart(disp_year_chart+disp_year_chart_line, use_container_width=True)
    st.write("Average displacement of cars has gone down along the years as we move forward to more fuel efficient vehicles which can give better performance with a smaller engine thus requiring less fuel for the same distance.")

with tab3:
    st.subheader("Check trends of other attributes over the years:")
    selected_column_toy = st.selectbox("Select attribute", continuous_attr, key=4)
    chart_line_check_toy = st.checkbox("Show Regression Line", key=5, value=True)
    if selected_column_toy:
        chart_toy = alt.Chart(cars_sample).mark_circle().encode(x=alt.X('year', scale=alt.Scale(domain=[1935, 2021])), y=selected_column_toy).interactive()
        chart_line_toy = chart_toy.transform_regression('year', selected_column_toy).mark_line(color='red')
        st.altair_chart(chart_toy+chart_line_toy if chart_line_check_toy else chart_toy, theme="streamlit", use_container_width=True)

# TAB 4
with tab4:
    st.subheader("Conclusion:")
    st.write("Modern cars have adopted smaller engines with time in pursuit of better fuel economy while not compromising on performance. Top speed and acceleration have shown continuous improvement over several decades. These advancements are the result of breakthroughs in various fields of Science and Engineering. Performance gains have been achieved through aerodynamics, electronic systems and electric motors, replacing the need for big, bulky engines.")
    st.write("In summary, this project has unveiled the remarkable evolution of cars over several decades, showcasing the consistent improvement in performance and the shift towards eco-friendly vehicles. It highlights the intersection of technological progress and environmental awareness in shaping the automotive landscape, suggesting a promising future where performance and sustainability coexist in the world of automobiles.")
    
