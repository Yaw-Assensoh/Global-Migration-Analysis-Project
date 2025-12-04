# utils/visualizations.py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_migration_leaders_chart(df, top_n=10):
    """Create migration leaders visualization"""
    top_imm = df.nlargest(top_n, 'Net_Migrants')
    top_emm = df.nsmallest(top_n, 'Net_Migrants')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Top {top_n} Immigration Countries', f'Top {top_n} Emigration Countries')
    )
    
    fig.add_trace(
        go.Bar(
            y=top_imm['Country'],
            x=top_imm['Net_Migrants'] / 1e6,
            orientation='h',
            marker_color='#2E86AB',
            text=[f'{x/1e6:.1f}M' for x in top_imm['Net_Migrants']],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            y=top_emm['Country'],
            x=abs(top_emm['Net_Migrants']) / 1e6,
            orientation='h',
            marker_color='#A23B72',
            text=[f'{abs(x)/1e6:.1f}M' for x in top_emm['Net_Migrants']],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Net Migrants (Millions)",
        yaxis_title="Country"
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap for demographic indicators"""
    numeric_cols = ['Net_Migrants', 'Migration_Rate_per_1000', 'Population', 'Density',
                   'Fertility_Rate', 'Median_Age', 'Urban_Pop_Percent', 'Yearly_Change']
    
    # Filter to existing columns
    existing_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[existing_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Correlation Matrix: Migration vs Demographics',
        width=700,
        height=700
    )
    
    return fig

def create_continent_analysis(df):
    """Create continent-level analysis visualization"""
    continent_stats = df.groupby('Continent').agg({
        'Net_Migrants': 'sum',
        'Population': 'sum',
        'Country': 'count',
        'Migration_Rate_per_1000': 'mean'
    }).round(2)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Net Migration', 'Migration Rate per 1000',
                       'Number of Countries', 'Total Population')
    )
    
    # Panel 1: Total Net Migration
    colors = ['#2E86AB' if x > 0 else '#A23B72' for x in continent_stats['Net_Migrants']]
    fig.add_trace(
        go.Bar(x=continent_stats.index, y=continent_stats['Net_Migrants']/1e6,
              marker_color=colors, text=[f'{x/1e6:.1f}M' for x in continent_stats['Net_Migrants']]),
        row=1, col=1
    )
    
    # Panel 2: Migration Rate
    fig.add_trace(
        go.Bar(x=continent_stats.index, y=continent_stats['Migration_Rate_per_1000'],
              marker_color='#F18F01', text=[f'{x:.1f}' for x in continent_stats['Migration_Rate_per_1000']]),
        row=1, col=2
    )
    
    # Panel 3: Number of Countries
    fig.add_trace(
        go.Bar(x=continent_stats.index, y=continent_stats['Country'],
              marker_color='#73AB84', text=continent_stats['Country']),
        row=2, col=1
    )
    
    # Panel 4: Total Population
    fig.add_trace(
        go.Bar(x=continent_stats.index, y=continent_stats['Population']/1e9,
              marker_color='#99C1B9', text=[f'{x/1e9:.1f}B' for x in continent_stats['Population']]),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig