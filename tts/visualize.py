import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from tkinter import ttk
from ipywidgets import interact, FloatSlider, Text, Box, Label, HBox, VBox, Layout, Output, interactive_output, GridspecLayout, Button, Dropdown
import numpy as np
from sklearn.compose import ColumnTransformer
from tts.basis import BSplineBasis
import mplcursors
from IPython.display import display
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import bokeh.plotting as bplt
from bokeh.io import output_notebook, push_notebook, show, curdoc
from bokeh.models import ColumnDataSource, HoverTool
import bokeh
from bokeh.events import Tap
from bisect import bisect_left, bisect_right
import pandas as pd
def simple_interactive_plot(f, time_horizon, trajectory_range, feature_ranges, n_points=1000, figsize=(8, 3)):
    """
    f: function of time t and features x that returns a trajectory y, signature f(t, **x)
    time_horizon: the time horizon of the trajectory
    trajectory_range: the range of the trajectory, tuple (y_min, y_max)
    feature_ranges: a dictionary of feature names and their ranges, e.g. {'a': (0, 1), 'b': (0, 2)}
    n_points: number of points to plot
    """
    t = np.linspace(0, time_horizon, n_points)

    # Set up the figure and axes
    fig = plt.figure(figsize=figsize)
    line, = plt.plot([], [], lw=2) # initialize the line with empty data

    plt.title("y = tts(t)")
    plt.xlim(0, time_horizon)
    plt.ylim(trajectory_range[0], trajectory_range[1])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.close()
    
    def plot_f(**x):
        y = f(t, **x)
        line.set_data(t, y)
        display(fig)
    
    ## Generate our user interface.
    # Create a dictionary of sliders, one for each feature.
    sliders = {}
    for k, v in feature_ranges.items():
        sliders[k] = FloatSlider(min=v[0], max=v[1], step=0.01, value=v[0])

    interact(plot_f, **sliders);


def simple_tts_plot(litmodel, dataset, trajectory_range, n_points=100, figsize=(8, 3)):
    """
    litmodel: a LitModel object
    dataset: a BaseDataset object
    trajectory_range: the range of the trajectory, tuple (y_min, y_max)
    n_points: number of points to plot
    """
    # Get the time horizon and trajectory range
    time_horizon = litmodel.config.T
    feature_names = dataset.get_feature_names()
    feature_ranges = dataset.get_feature_ranges()

    def trajectory(t, **x):
        features = np.array([x[feature_name] for feature_name in feature_names])
        return litmodel.model.forecast_trajectory(features,t)
    
    simple_interactive_plot(trajectory, time_horizon, trajectory_range, feature_ranges, n_points, figsize=figsize)


def draw_rectangles(transitions, colors):

    total_width = 5
    total_height = 1

    widths = np.diff(transitions)
    widths = widths * (total_width / np.sum(widths))

    fig, ax = plt.subplots(figsize=(2.75, 0.2))
    # colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    x = 0

    for i in range(len(widths)):
        rect_width = widths[i]
        rect = plt.Rectangle((x, 0), rect_width, total_height, color=colors[i], alpha=1.0, linewidth=0.0)
        ax.add_patch(rect)
        x += rect_width

    plt.xlim(0, total_width)
    plt.ylim(0, total_height)
    plt.axis('off')
    return fig

def get_meta_template(litmodel,features,index,query_points):

    config = litmodel.config
    bspline = BSplineBasis(config.n_basis, (0,config.T), internal_knots=config.internal_knots)

    # Prepare features
    features = np.tile(features, (len(query_points), 1))
    features[:,index] = query_points

    # Get the coefficients
    coeffs = litmodel.model.predict_latent_variables(features)

    templates = [bspline.get_template_from_coeffs(coeffs[i,:])[0] for i in range(len(query_points))]

    # Combine templates
    combined_templates = []
    transitions = []

    combined_templates.append(templates[0])
    transitions.append(query_points[0])
    for i in range(1,len(query_points)):
        if templates[i] != templates[i-1]:
            combined_templates.append(templates[i])
            transitions.append(query_points[i])
    transitions.append(query_points[-1])

    return combined_templates, transitions

class MetaTemplateContext():

    def __init__(self, litmodel, feature_names, feature_ranges, feature_types, column_transformer=None, n_points=100):
        self.litmodel = litmodel
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges
        self.feature_types = feature_types
        self.n_points = n_points
        self.combined_templates = []
        self.transitions = []
        self.current_template = []
        self.current_transition_points = []
        self.config = litmodel.config
        self.bspline = BSplineBasis(self.config.n_basis, (0,self.config.T), internal_knots=self.config.internal_knots)
        self.transition_points_trajectories = {}
        self.column_transformer = column_transformer
    
    def update(self, raw_features):
        """
        Args:
            raw_features: a pandas DataFrame of raw features (not transformed, containing only one row)
        """
        self.current_raw_features = raw_features
        self.current_transformed_features = self.column_transformer.transform(raw_features) if self.column_transformer else raw_features.values # This is 2D
        self.combined_templates = []
        self.transitions = []
        for feature_name in self.feature_names:
            combined_templates, transitions = self._get_meta_template(raw_features,feature_name)
            self.combined_templates.append(combined_templates)
            self.transitions.append(transitions)

        coeffs = self.litmodel.model.predict_latent_variables(self.current_transformed_features)
        template, transition_points = self.bspline.get_template_from_coeffs(coeffs[0,:])

        new_template = False
        if template != self.current_template:
            new_template = True
            # Update the transition point trajectories
            self.transition_points_trajectories = {}
            for x_axis in self.feature_names:
                query_points, trans_x_all, trans_y_all = self._get_transition_point_curves(raw_features, x_axis)
                self.transition_points_trajectories[x_axis] = {
                    'query_points': query_points,
                    'transition_points': [
                        {'t':trans_x_all[index],
                         'y':trans_y_all[index]}
                    for index in range(len(trans_x_all))]
                }
        self.current_template = template
        self.current_transition_points = transition_points
        return new_template
    
    def _get_query_points(self, feature_name):
        if self.feature_types[feature_name] == 'continuous':
            return np.linspace(self.feature_ranges[feature_name][0],self.feature_ranges[feature_name][1],self.n_points)
        elif self.feature_types[feature_name] == 'categorical' or self.feature_types[feature_name] == 'binary':
            return np.array(self.feature_ranges[feature_name])

    def _get_meta_template(self, raw_features, feature_name):

        query_points = self._get_query_points(feature_name)
       
        # Prepare features
        new_features = pd.concat([raw_features]*len(query_points), axis=0, ignore_index=True)
        new_features[feature_name] = query_points

        # Transform features
        transformed_features = self.column_transformer.transform(new_features)

        # Get the coefficients
        coeffs = self.litmodel.model.predict_latent_variables(transformed_features)

        templates = [self.bspline.get_template_from_coeffs(coeffs[i,:])[0] for i in range(len(query_points))]

        # Combine templates
        combined_templates = []
        transitions = []

        combined_templates.append(templates[0])
        transitions.append(query_points[0])
        for i in range(1,len(query_points)):
            if templates[i] != templates[i-1]:
                combined_templates.append(templates[i])
                transitions.append(query_points[i])
        transitions.append(query_points[-1])

        # If the feature is categorical or binary, the transition points are the categories.
        return combined_templates, transitions
    
    def _get_transition_point_curves(self, raw_features, x_axis):
        # Get the index of x_axis in feature_names
        x_axis_index = self.feature_names.index(x_axis)
        combined_templates, transitions = self.get_meta_templates_and_transitions(x_axis_index)
        curr_x_axis_value = raw_features.loc[0,x_axis]

        if self.feature_types[x_axis] == 'categorical' or self.feature_types[x_axis] == 'binary':
            numerical_transitions = [self.feature_ranges[x_axis].index(transition) for transition in transitions]
            curr_x_axis_value_numerical = self.feature_ranges[x_axis].index(curr_x_axis_value)
            first_transition_point_on_the_right = bisect_right(numerical_transitions, curr_x_axis_value_numerical)
            prev_point = numerical_transitions[first_transition_point_on_the_right-1]
            next_point = numerical_transitions[first_transition_point_on_the_right]
            query_points = [self.feature_ranges[x_axis][i] for i in range(prev_point,next_point)]

        elif self.feature_types[x_axis] == 'continuous':
            first_transition_point_on_the_right = bisect_left(transitions, curr_x_axis_value)
        
            if first_transition_point_on_the_right == 0:
                x_axis_range = (transitions[0], transitions[1])
            else:
                x_axis_range = (transitions[first_transition_point_on_the_right-1], transitions[first_transition_point_on_the_right])
            eps = (self.feature_ranges[x_axis][1] - self.feature_ranges[x_axis][0]) / self.n_points
            query_points = np.linspace(x_axis_range[0] + eps,x_axis_range[1] - eps, self.n_points)

        # Prepare features
        new_features = pd.concat([raw_features]*len(query_points), axis=0, ignore_index=True)
        new_features[x_axis] = query_points

        # Transform features
        features_tiled = self.column_transformer.transform(new_features)

        # Get the coefficients
        coeffs = self.litmodel.model.predict_latent_variables(features_tiled)

        
        trans_x_coords_combined = [self.bspline.get_template_from_coeffs(coeffs[i,:])[1] for i in range(len(query_points))]
        n_transitions = len(trans_x_coords_combined[0])


        trans_x_coords_all = [np.array([trans_x_coords_combined[q][index] for q in range(len(query_points))]) for index in range(n_transitions)]

        # Calculate y-coordinates of the transition point
        trans_y_coords_all = [np.diagonal(self.litmodel.model.forecast_trajectories(features_tiled,trans_x_coords_all[index])) for index in range(n_transitions)]
        
        return query_points, trans_x_coords_all, trans_y_coords_all


    def get_meta_templates_and_transitions(self, index):
        return self.combined_templates[index], self.transitions[index]
    
    def get_transition_point_curve(self, x_axis, y_axis, index):
        return self.transition_points_trajectories[x_axis]['query_points'], self.transition_points_trajectories[x_axis]['transition_points'][index][y_axis]
    
    def get_current_value(self, x_axis):
        return self.current_raw_features.loc[0,x_axis]
    
    def get_current_trajectory(self):
        t = np.linspace(0, self.litmodel.config.T, self.n_points)
        y = self.litmodel.model.forecast_trajectory(self.current_transformed_features[0,:],t)
        return t, y

    def get_current_transition_points(self):
        transition_points_x = np.array(self.current_transition_points)
        transition_points_y = self.litmodel.model.forecast_trajectory(self.current_transformed_features[0,:], transition_points_x)
        return transition_points_x, transition_points_y



def random_color():
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    for i in range(9):
        yield CB_color_cycle[i]
    for i in range(1000):
        rng = np.random.default_rng(i)
        r, g, b = rng.random(3)
        yield (r, g, b)

def advanced_tts_plot(litmodel, dataset, trajectory_range, n_points=100, figsize=(8, 3)):
    
    # Get the time horizon and trajectory range
    time_horizon = litmodel.config.T
    feature_names = dataset.get_feature_names()
    feature_ranges = dataset.get_feature_ranges()

    config = litmodel.config

    bspline = BSplineBasis(config.n_basis, (0,config.T), internal_knots=config.internal_knots)

    bands = [Output() for _ in range(len(feature_names))]
    main_plot = Output()

    t = np.linspace(0, time_horizon, n_points)

    # Set up the figure and axes
    main_fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot([], [], lw=2, zorder=1) # initialize the line with empty data
    scat = ax.scatter([], [], s=20, c='black', zorder=2, picker=True)



    def on_pick(event):
        with main_plot:
            # ax.text(0.5,1,"Text")
            # plt.draw()
            main_plot.clear_output(wait=True)
            print("Hello")
            # line.set_linestyle("--")
            # display(main_fig)

    # cursor = mplcursors.cursor(scat, hover=False)
    # cursor.connect("add", on_pick)
    main_fig.canvas.mpl_connect('pick_event', on_pick)


    
    plt.title("y = tts(t)")
    plt.xlim(0, time_horizon)
    plt.ylim(trajectory_range[0], trajectory_range[1])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.close()

    random_color_iter = iter(random_color())

    color_map = {}

    
    def update(**x):
        features = np.array([x[feature_name] for feature_name in feature_names])
        coeffs = litmodel.model.predict_latent_variables(features.reshape(1,-1))
        template, transition_points = bspline.get_template_from_coeffs(coeffs[0,:])
        
        
        for i in range(len(feature_names)):
            combined_templates, transitions = get_meta_template(litmodel,features,i,np.linspace(feature_ranges[feature_names[i]][0],feature_ranges[feature_names[i]][1],100))
            colors_to_use = []
            for combined_template in combined_templates:
                if tuple(combined_template) not in color_map:
                    color_map[tuple(combined_template)] = next(random_color_iter)
                colors_to_use.append(color_map[tuple(combined_template)])
            output = bands[i]
            with output:
                output.clear_output(wait=True)
                fig = draw_rectangles(transitions, colors_to_use)
                display(fig)

        y = litmodel.model.forecast_trajectory(features,t)
        transition_points_y = litmodel.model.forecast_trajectory(features,np.array(transition_points))
        with main_plot:
            main_plot.clear_output(wait=True)
            line.set_data(t, y)
            scat.set_offsets(np.c_[transition_points, transition_points_y])
            display(main_fig)
   
    
    ## Generate our user interface.
    # Create a dictionary of sliders, one for each feature.
    sliders = {}
    for k, v in feature_ranges.items():
        sliders[k] = FloatSlider(min=v[0], max=v[1], step=0.01, value=v[0])
        sliders[k].layout.margin = '0px 0px 0px 5px'

    grid = GridspecLayout(len(feature_names)*2, 8)
    grid.layout.height = f'{len(feature_names)*75}px'
    # grid.layout.grid_template_columns = '10px auto'

    for i, feature_name in enumerate(feature_names):
        label = Label(value=feature_name)
        label.layout.width = '50px'
        label.layout.align_self = 'center'
        grid[2*i:2*i+1,0] = label
        grid[2*i,1:8] = sliders[feature_name]
        grid[2*i+1,1:8] = bands[i]

    submit_button = Button(
        description='Submit',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click to submit'
    )

    submit_button.on_click(on_pick)


    layout = HBox([grid,main_plot, submit_button])

    # # Display the layout
    display(layout)

    interactive_output(update, sliders);


def _verify_column_transformer(column_transformer, feature_names):
    if column_transformer is None:
        return True
    elif isinstance(column_transformer, ColumnTransformer):
        transformers = column_transformer.transformers_
        all_columns = []
        for name, fitted_transformer, columns in transformers:
            if isinstance(columns, str):
                all_columns.append(columns)
            elif isinstance(columns, int):
                all_columns.append(feature_names[columns])
            elif isinstance(columns, list):
                for column in columns:
                    if isinstance(column, int):
                        all_columns.append(feature_names[column])
                    elif isinstance(column, str):
                        all_columns.append(column)
                    else:
                        raise ValueError(f"Invalid column type {type(column)}")
        if all_columns == feature_names:
            return True
        else:
            return False        
    else:
        return False
    
def _transform_feature_dict(x, feature_names, column_transformer):
    features = np.array([x[feature_name] for feature_name in feature_names])
    if column_transformer is None:
        return features, features
    else:
        return features, column_transformer.transform(features.reshape(1,-1))[0,:]

def _extract_raw_features(x, feature_names):
    # This function takes a dictionary of features and returns a dataframe with one row and the features as columns
    return pd.DataFrame({feature_name: x[feature_name] for feature_name in feature_names}, index=[0])

def expert_tts_plot(litmodel, dataset, trajectory_range, n_points=100, figsize=(8, 3), column_transformer=None):

    # Get the time horizon and trajectory range
    time_horizon = litmodel.config.T
    feature_names = dataset.get_feature_names()
    feature_ranges = dataset.get_feature_ranges()
    feature_types = dataset.get_feature_types()

    if not _verify_column_transformer(column_transformer, feature_names):
        print("Warning: column transformer does not match feature names")
        raise ValueError("Invalid column transformer")

    config = litmodel.config

    bspline = BSplineBasis(config.n_basis, (0,config.T), internal_knots=config.internal_knots)

    bands = [Output() for _ in range(len(feature_names))]
    main_plot = Output()
    secondary_plot = Output()

    t = np.linspace(0, time_horizon, n_points)

    # Set up the figure and axes
    main_fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot([], [], lw=2, zorder=1) # initialize the line with empty data
    scat = ax.scatter([], [], s=20, c='black', zorder=2, picker=True)
    plt.title("y = tts(t)")
    plt.xlim(0, time_horizon)
    plt.ylim(trajectory_range[0], trajectory_range[1])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.close()

    second_fig, second_ax = plt.subplots(figsize=(figsize[0]*0.65, figsize[1]*0.65))
    second_line, = second_ax.plot([],[],lw=2, zorder=1)
    second_scat = second_ax.scatter([],[], s=30, c='black', zorder=2)
    plt.close()



    transition_point_dropdown = Dropdown(
    options = ['?'],
    value = '?',
    description = 'Transition point',
    disabled=False,
    layout=Layout(width='150px'),
    style={'description_width': 'auto'}
    )

    second_grid = GridspecLayout(7, 6)
    second_grid.layout.height = '350px'

    y_axis_dropdown = Dropdown(
        options = ['t','y'],
        value = 'y',
        description = '',
        disabled=False,
        layout=Layout(width='50px'),
        style={'description_width': 'auto'}
    )
    x_axis_dropdown = Dropdown(
        options = feature_names,
        value = feature_names[0],
        description = '',
        disabled=False,
        layout=Layout(width='150px'),
        style={'description_width': 'auto'}
    )

    # Create two empty box widgets to act as spacers
    spacer_box1 = Box(layout=Layout(flex="1 1 auto"))
    spacer_box2 = Box(layout=Layout(flex="1 1 auto"))

    # Create an HBox containing the spacers and the button
    centered_hbox = HBox([spacer_box1, x_axis_dropdown, spacer_box2])
    centered_hbox_2 = HBox([spacer_box1, transition_point_dropdown, spacer_box2])

    second_grid[3,0] = y_axis_dropdown
    second_grid[6,1:6] = centered_hbox
    second_grid[0,1:6] = centered_hbox_2
    second_grid[1:6,1:6] = secondary_plot

    ## Generate our user interface.
    # Create a dictionary of sliders, one for each feature.
    sliders = {}
    for k, v in feature_ranges.items():
        sliders[k] = FloatSlider(min=v[0], max=v[1], step=0.01, value=v[0])
        sliders[k].layout.margin = '0px 0px 0px 5px'



    random_color_iter = iter(random_color())
    color_map = {}
    meta_template_context = MetaTemplateContext(litmodel, feature_names, feature_ranges, feature_types, column_transformer=column_transformer, n_points=n_points)


    def update(**x):
        raw_features =  _extract_raw_features(x, feature_names)

        new_template = meta_template_context.update(raw_features)

        for i in range(len(feature_names)):
            combined_templates, transitions = meta_template_context.get_meta_templates_and_transitions(i)
            colors_to_use = []
            for combined_template in combined_templates:
                if tuple(combined_template) not in color_map:
                    color_map[tuple(combined_template)] = next(random_color_iter)
                colors_to_use.append(color_map[tuple(combined_template)])
            output = bands[i]
            with output:
                output.clear_output(wait=True)
                fig = draw_rectangles(transitions, colors_to_use)
                display(fig)
        
        if new_template:
            num_transition_points = len(meta_template_context.current_transition_points)
            transition_point_dropdown.options = ['?'] + list(range(1,num_transition_points+1))
            transition_point_dropdown.value = '?'
        else:
            update_secondary(transition_point=transition_point_dropdown.value, x_axis=x_axis_dropdown.value, y_axis=y_axis_dropdown.value)

        t, y = meta_template_context.get_current_trajectory()
        transition_points_x, transition_points_y = meta_template_context.get_current_transition_points()

        with main_plot:
            main_plot.clear_output(wait=True)
            line.set_data(t, y)
            scat.set_offsets(np.c_[transition_points_x, transition_points_y])
            display(main_fig)

        
    def update_secondary(**x):
        if x['transition_point'] == '?':
            transition_point = None
        else:
            transition_point = x['transition_point'] -1
           
        with secondary_plot:
           
            if transition_point is not None:
                x_axis = x['x_axis']
                y_axis = x['y_axis']
                x_values, y_values = meta_template_context.get_transition_point_curve(x_axis, y_axis, transition_point)
               
                # print(x_values)
                # print(y_values)
                second_ax.set_xlim(x_values[0], x_values[-1])
                y_max = np.max(y_values)
                y_min = np.min(y_values)
                if y_max == y_min:
                    y_min = y_min - 1.0
                    y_max = y_max + 1.0
                second_ax.set_ylim(y_min, y_max)
                curr_x = meta_template_context.get_current_value(x_axis)
                if curr_x > x_values[-1]:
                    curr_x = x_values[-1]
                elif curr_x < x_values[0]:
                    curr_x = x_values[0]

                curr_y = y_values[bisect_left(x_values, curr_x)]
                
                secondary_plot.clear_output(wait=True)
                second_line.set_data(x_values, y_values)
                second_scat.set_offsets(np.c_[[curr_x],[curr_y]])
                display(second_fig)
            else:
                secondary_plot.clear_output()
                

    grid = GridspecLayout(len(feature_names)*2, 8)
    grid.layout.height = f'{len(feature_names)*75}px'
    # grid.layout.grid_template_columns = '10px auto'

    for i, feature_name in enumerate(feature_names):
        label = Label(value=feature_name)
        label.layout.width = '50px'
        label.layout.align_self = 'center'
        grid[2*i:2*i+1,0] = label
        grid[2*i,1:8] = sliders[feature_name]
        grid[2*i+1,1:8] = bands[i]




    layout = HBox([grid,main_plot, second_grid])

    # # Display the layout
    display(layout)

    interactive_output(update, sliders);
    interactive_output(update_secondary, {'transition_point':transition_point_dropdown, 'x_axis':x_axis_dropdown, 'y_axis':y_axis_dropdown})


class FigureContext(Output):
    def __init__(self, fig):
        output_notebook(hide_banner=True)
        super().__init__()
        self._figure = fig
        self.observe(lambda event: self.set_handle(), names="_view_count")

        
    # def _call_widget_constructed(widget):
    #     super()._call_widget_constructed(widget)
    #     widget.set_handle()

    def set_handle(self):
        self.clear_output()
        with self:
            self._handle = show(self._figure, notebook_handle=True)
            
    def get_handle(self):
        return self._handle
    
    def get_figure(self):
        return self._figure
    
    def update(self):
        push_notebook(handle=self._handle)

    
    def figure(*args, **kwargs):
        fig = bplt.figure(*args, **kwargs)
        return FigureContext(fig)


def bokeh_tts_plot(litmodel, dataset, trajectory_range, n_points=100, plot_size=(800, 300)):
    
    # Get the time horizon and trajectory range
    time_horizon = litmodel.config.T
    feature_names = dataset.get_feature_names()
    feature_ranges = dataset.get_feature_ranges()

    config = litmodel.config

    bspline = BSplineBasis(config.n_basis, (0,config.T), internal_knots=config.internal_knots)

    bands = [Output() for _ in range(len(feature_names))]
    main_plot = Output()

    t = np.linspace(0, time_horizon, n_points)

    # Set up the figure and axes


    main_fig_context = FigureContext.figure(title="y = tts(t)", x_range=(0, time_horizon), y_range=(trajectory_range[0], trajectory_range[1]),
                  x_axis_label='t', y_axis_label='y', width=plot_size[0], height=plot_size[1], tools="pan,box_zoom,reset,save,wheel_zoom")
    main_fig_context._view_count = 0
    line_source = ColumnDataSource(data=dict(x=[], y=[]))
    scatter_source = ColumnDataSource(data=dict(x=[], y=[]))

    main_fig_context.get_figure().line('x', 'y', source=line_source, line_width=2)
    main_fig_context.get_figure().scatter('x', 'y', source=scatter_source, size=10, color='black')

    tooltips = [('x', '@x'), ('y', '@y')]

    main_fig_context.get_figure().add_tools(HoverTool(tooltips=tooltips, renderers=[main_fig_context.get_figure().renderers[1]]))
    




    # cursor = mplcursors.cursor(scat, hover=False)
    # cursor.connect("add", on_pick)
    # main_fig.canvas.mpl_connect('pick_event', on_pick)


    
    # plt.title("y = tts(t)")
    # plt.xlim(0, time_horizon)
    # plt.ylim(trajectory_range[0], trajectory_range[1])
    # plt.xlabel('t')
    # plt.ylabel('y')
    # plt.close()

    random_color_iter = iter(random_color())

    color_map = {}

    

   
    
    ## Generate our user interface.
    # Create a dictionary of sliders, one for each feature.
    sliders = {}
    for k, v in feature_ranges.items():
        sliders[k] = FloatSlider(min=v[0], max=v[1], step=0.01, value=v[0])
        sliders[k].layout.margin = '0px 0px 0px 5px'

    grid = GridspecLayout(len(feature_names)*2, 8)
    grid.layout.height = f'{len(feature_names)*75}px'
    # grid.layout.grid_template_columns = '10px auto'

    for i, feature_name in enumerate(feature_names):
        label = Label(value=feature_name)
        label.layout.width = '50px'
        label.layout.align_self = 'center'
        grid[2*i:2*i+1,0] = label
        grid[2*i,1:8] = sliders[feature_name]
        grid[2*i+1,1:8] = bands[i]

    submit_button = Button(
        description='Submit',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click to submit'
    )

    text_output = Output()

    def on_pick(event):
        with text_output:
            print("Hello")

    submit_button.on_click(on_pick)



    layout = HBox([grid,main_fig_context, submit_button, text_output])

    
    # Display the layout
    display(layout)

    # main_fig_context.set_handle()


    # with main_plot:
    #     output_notebook(bokeh.resources.INLINE)
    #     main_plot_handle = show(main_fig, notebook_handle=True)

    def on_point_pick(event):
        coords = (event.x, event.y)
        on_pick(event)

    main_fig_context.get_figure().on_event(Tap, on_point_pick)





    def update(**x):
        features = np.array([x[feature_name] for feature_name in feature_names])
        coeffs = litmodel.model.predict_latent_variables(features.reshape(1,-1))
        template, transition_points = bspline.get_template_from_coeffs(coeffs[0,:])
        
        
        for i in range(len(feature_names)):
            combined_templates, transitions = get_meta_template(litmodel,features,i,np.linspace(feature_ranges[feature_names[i]][0],feature_ranges[feature_names[i]][1],100))
            colors_to_use = []
            for combined_template in combined_templates:
                if tuple(combined_template) not in color_map:
                    color_map[tuple(combined_template)] = next(random_color_iter)
                colors_to_use.append(color_map[tuple(combined_template)])
            output = bands[i]
            with output:
                output.clear_output(wait=True)
                fig = draw_rectangles(transitions, colors_to_use)
                display(fig)

        y = litmodel.model.forecast_trajectory(features,t)
        transition_points_y = litmodel.model.forecast_trajectory(features,np.array(transition_points))
        
        # main_plot.clear_output(wait=True)
        rend = main_fig_context.get_figure().renderers
        # with text_output:
        #     text_output.clear_output(wait=True)
        #     print(rend)
        rend[0].data_source.data = {'x': t, 'y': y}
        rend[1].data_source.data = {'x': transition_points, 'y': transition_points_y}

        main_fig_context.set_handle()

    interactive_output(update, sliders)


def return_advanced_tts_plot(litmodel, dataset, trajectory_range, n_points=100, figsize=(8, 3)):
    
    # Get the time horizon and trajectory range
    time_horizon = litmodel.config.T
    feature_names = dataset.get_feature_names()
    feature_ranges = dataset.get_feature_ranges()

    config = litmodel.config

    bspline = BSplineBasis(config.n_basis, (0,config.T), internal_knots=config.internal_knots)

    bands = [Output() for _ in range(len(feature_names))]
    main_plot = Output()

    t = np.linspace(0, time_horizon, n_points)

    # Set up the figure and axes
    main_fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot([], [], lw=2, zorder=1) # initialize the line with empty data
    scat = ax.scatter([], [], s=20, c='black', zorder=2, picker=True)



    def on_pick(event):
        with main_plot:
            # ax.text(0.5,1,"Text")
            # plt.draw()
            main_plot.clear_output(wait=True)
            print("Hello")
            # line.set_linestyle("--")
            # display(main_fig)

    # cursor = mplcursors.cursor(scat, hover=False)
    # cursor.connect("add", on_pick)
    main_fig.canvas.mpl_connect('pick_event', on_pick)


    
    plt.title("y = tts(t)")
    plt.xlim(0, time_horizon)
    plt.ylim(trajectory_range[0], trajectory_range[1])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.close()

    random_color_iter = iter(random_color())

    color_map = {}

    
    def update(**x):
        features = np.array([x[feature_name] for feature_name in feature_names])
        coeffs = litmodel.model.predict_latent_variables(features.reshape(1,-1))
        template, transition_points = bspline.get_template_from_coeffs(coeffs[0,:])
        
        
        for i in range(len(feature_names)):
            combined_templates, transitions = get_meta_template(litmodel,features,i,np.linspace(feature_ranges[feature_names[i]][0],feature_ranges[feature_names[i]][1],100))
            colors_to_use = []
            for combined_template in combined_templates:
                if tuple(combined_template) not in color_map:
                    color_map[tuple(combined_template)] = next(random_color_iter)
                colors_to_use.append(color_map[tuple(combined_template)])
            output = bands[i]
            with output:
                output.clear_output(wait=True)
                fig = draw_rectangles(transitions, colors_to_use)
                display(fig)

        y = litmodel.model.forecast_trajectory(features,t)
        transition_points_y = litmodel.model.forecast_trajectory(features,np.array(transition_points))
        with main_plot:
            main_plot.clear_output(wait=True)
            line.set_data(t, y)
            scat.set_offsets(np.c_[transition_points, transition_points_y])
            display(main_fig)
   
    
    ## Generate our user interface.
    # Create a dictionary of sliders, one for each feature.
    sliders = {}
    for k, v in feature_ranges.items():
        sliders[k] = FloatSlider(min=v[0], max=v[1], step=0.01, value=v[0])
        sliders[k].layout.margin = '0px 0px 0px 5px'

    grid = GridspecLayout(len(feature_names)*2, 8)
    grid.layout.height = f'{len(feature_names)*75}px'
    # grid.layout.grid_template_columns = '10px auto'

    for i, feature_name in enumerate(feature_names):
        label = Label(value=feature_name)
        label.layout.width = '50px'
        label.layout.align_self = 'center'
        grid[2*i:2*i+1,0] = label
        grid[2*i,1:8] = sliders[feature_name]
        grid[2*i+1,1:8] = bands[i]

    submit_button = Button(
        description='Submit',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click to submit'
    )

    submit_button.on_click(on_pick)


    layout = HBox([grid,main_plot, submit_button])

    # # Display the layout
    display(layout)

    interactive_output(update, sliders);


# def standalone_tts_plot(litmodel, dataset, trajectory_range, n_points=100, figsize=(8, 3)):
#     matplotlib.use('TkAgg')
#     time_horizon = litmodel.config.T
#     feature_names = dataset.get_feature_names()
#     feature_ranges = dataset.get_feature_ranges()

#     config = litmodel.config

#     bspline = BSplineBasis(config.n_basis, (0,config.T), internal_knots=config.internal_knots)

#     t = np.linspace(0, time_horizon, n_points)

#     main_fig, ax = plt.subplots(figsize=figsize)
#     line, = ax.plot([], [], lw=2, zorder=1)
#     scat = ax.scatter([], [], s=20, c='black', zorder=2, picker=True)

#     plt.title("y = tts(t)")
#     plt.xlim(0, time_horizon)
#     plt.ylim(trajectory_range[0], trajectory_range[1])
#     plt.xlabel('t')
#     plt.ylabel('y')

#     random_color_iter = iter(random_color())

#     color_map = {}

#     def update(event=None):
#         x = {feature_name: sliders[feature_name].get() for feature_name in feature_names}
#         features = np.array([x[feature_name] for feature_name in feature_names])
#         coeffs = litmodel.model.predict_latent_variables(features.reshape(1,-1))
#         template, transition_points = bspline.get_template_from_coeffs(coeffs[0,:])

#         y = litmodel.model.forecast_trajectory(features,t)
#         transition_points_y = litmodel.model.forecast_trajectory(features,np.array(transition_points))
#         line.set_data(t, y)
#         scat.set_offsets(np.c_[transition_points, transition_points_y])

#         for i in range(len(feature_names)):
#             combined_templates, transitions = get_meta_template(litmodel, features, i, np.linspace(feature_ranges[feature_names[i]][0], feature_ranges[feature_names[i]][1], 100))
#             colors_to_use = []
#             for combined_template in combined_templates:
#                 if tuple(combined_template) not in color_map:
#                     color_map[tuple(combined_template)] = next(random_color_iter)
#                 colors_to_use.append(color_map[tuple(combined_template)])
#             fig = draw_rectangles(transitions, colors_to_use)
#             bands_canvases[i].figure = fig
#             bands_canvases[i].draw()

#         canvas.draw()

#     # Create a Tkinter window
#     window = tk.Tk()
#     window.title("Interactive Plot")

#     # Create a frame for sliders and labels
#     sliders_frame = ttk.Frame(window)
#     sliders_frame.grid(column=0, row=0, sticky="W")

#     sliders = {}
#     bands_canvases = []

#     for idx, feature_name in enumerate(feature_names):
#         label = ttk.Label(sliders_frame, text=feature_name)
#         label.grid(column=0, row=2 * idx, sticky="W")

#         slider = ttk.Scale(sliders_frame, from_=feature_ranges[feature_name][0], to=feature_ranges[feature_name][1], orient="horizontal", command=update)
#         slider.set(feature_ranges[feature_name][0])
#         slider.grid(column=1, row=2 * idx, sticky="EW")

#         sliders[feature_name] = slider

#         bands_fig = plt.figure(figsize=(6, 0.5))
#         bands_canvas = FigureCanvasTkAgg(bands_fig, master=sliders_frame)
#         bands_canvas.get_tk_widget().grid(column=1, row=2 * idx + 1, sticky="EW", padx=5, pady=5)
#         bands_canvases.append(bands_canvas)

#     # Create a canvas for the interactive plot
#     canvas = FigureCanvasTkAgg(main_fig, master=window)
#     canvas.get_tk_widget().grid(column=2, row=0, rowspan=2 * len(feature_names), padx=10)

#     # Create the Submit button
#     submit_button = ttk.Button(window, text="Submit", command=update)
#     submit_button.grid(column=3, row=2 * len(feature_names) - 1, sticky="E", pady=5)

#     # Run the Tkinter main loop
#     window.mainloop()
    

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from experiments.datasets import load_dataset
    from tts.lit_module import load_model
    import numpy as np


    litmodel = load_model('2023-04-28T12-52-14',benchmarks_folder='../experiments/benchmarks')
    dataset = load_dataset('synthetic_tumor_wilkerson_1', '../experiments/dataset_descriptions')
    standalone_tts_plot(litmodel, dataset, (0,2), n_points=100, figsize=(4,3))