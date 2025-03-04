import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout='wide')

st.title("Pokemon Ultimate Dataset")

file = "EDA_DATA-PROFILING/pokemon.csv"
data = pd.read_csv(file)

def plot_pokemon_damage_multiplier(pokemon_identifier):
    """
    Generates an interactive heatmap showing the damage multipliers of a selected Pokémon.
    The identifier can be either the Pokémon name or its Pokédex number.
    """
    # Search for the Pokémon
    if isinstance(pokemon_identifier, int):  # If input is a number, search by Pokedex number
        pokemon_data = data[data['pokedex_number'] == pokemon_identifier]
    else:  # If input is a name, search by name
        pokemon_data = data[data['name'].str.lower() == pokemon_identifier.lower()]
    
    # If no match found, return
    if pokemon_data.empty:
        st.error("Pokémon not found! Please check the name or Pokédex number.")
        return
    
    # Extract relevant columns (damage multipliers)
    effectiveness_cols = [col for col in data.columns if col.startswith("against_")]
    effectiveness_data = pokemon_data[effectiveness_cols].iloc[0]  # Get the first row

    # Sort the values from lowest to highest
    sorted_effectiveness = effectiveness_data.sort_values()

    # Convert to DataFrame for Plotly
    heatmap_data = pd.DataFrame({
        "Type": sorted_effectiveness.index,
        "Multiplier": sorted_effectiveness.values
    })

    # Create interactive heatmap
    fig = px.imshow(
        [heatmap_data["Multiplier"].values],
        labels=dict(x="Type", y=" ", color="Multiplier"),
        x=heatmap_data["Type"],
        y=["Damage Multiplier"],
        color_continuous_scale="RdBu_r",
        text_auto=True
    )

    # Update layout for better visualization
    fig.update_layout(
        title=f"Damage Multiplier Heatmap for {pokemon_data['name'].values[0]} (#{pokemon_identifier})",
        xaxis=dict(tickangle=-45),
        yaxis=dict(showticklabels=False),
        width=800, height=300
    )

    # Show plot
    st.plotly_chart(fig, use_container_width=True)

new_col_order = ['pokedex_number', 'name','japanese_name','type1', 'type2', 'hp', 'attack','defense', 'sp_attack', 'sp_defense', 'speed', 'abilities','height_m', 'weight_kg', 'generation', 'base_egg_steps', 'base_happiness', 'base_total', 'capture_rate',
       'classfication' , 'experience_growth','percentage_male', 'against_bug', 'against_dark', 'against_dragon',
       'against_electric', 'against_fairy', 'against_fight', 'against_fire',
       'against_flying', 'against_ghost', 'against_grass', 'against_ground',
       'against_ice', 'against_normal', 'against_poison', 'against_psychic',
       'against_rock', 'against_steel', 'against_water']

data = data[new_col_order]

search = st.text_input("Search Pokemon or Pokemon data:")

if search:
    filtered_df = data[data.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)]
else:
    filtered_df = data

sort_by = st.selectbox('Sort by Column', filtered_df.columns)
order = st.radio('Sort Order:', ['Ascending', 'Descending'])

ascending = True if order == 'Ascending' else False
sorted_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

st.dataframe(sorted_df)

st.title("Pokémon Type Effectiveness Heatmap (Interactive)")

# Get unique Pokémon names and numbers
pokemon_names = data[['pokedex_number', 'name']].drop_duplicates()

# Create selection inputs
selected_option = st.radio("Search by:", ["Name", "Pokedex Number"])

if selected_option == "Name":
    selected_name = st.selectbox("Select a Pokémon:", pokemon_names['name'].unique())
    if st.button("Show Heatmap"):
        st.dataframe(data[data['name'] == selected_name])
        plot_pokemon_damage_multiplier(selected_name)
else:
    selected_number = st.selectbox("Select a Pokédex Number:", pokemon_names['pokedex_number'].unique())
    if st.button("Show Heatmap"):
        plot_pokemon_damage_multiplier(selected_number)

