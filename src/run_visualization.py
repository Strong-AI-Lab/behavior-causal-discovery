
import argparse
import os
import time
import tqdm
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as colormaps
import matplotlib.patches as patches

from data.structure.chronology import Chronology


DEFAULT_SAVE_FOLDER = "data/anim"


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('structure', type=str, help='Load the model from a save folder.')
parser.add_argument('--save_folder', type=str, default=DEFAULT_SAVE_FOLDER, help=f'If specified, saves the animation to the specified folder. If not specified, the animation is saved to {DEFAULT_SAVE_FOLDER}.')
parser.add_argument('--save_name', type=str, default=None, help='If specified, saves the animation to the specified file. If not specified, the animation is saved to a file with the current time.')
parser.add_argument('--max_t', type=int, default=None, help='If specified, only runs the animation for the specified number of timesteps.')
parser.add_argument('--show_neighbours', action='store_true', help='If specified, shows the neighbours of each agent.')
parser.add_argument('--show_vision_field', action='store_true', help='If specified, shows the estimated vision field of each agent.')
parser.add_argument('--fps', type=int, default=12, help='Frames per second of the animation.')
args = parser.parse_args()



# Load structure
chronology = Chronology.deserialize(args.structure)
minX, maxX, minY, maxY = 0, 1, 0, 1
if args.max_t is not None:
    maxT = min(args.max_t, len(chronology.snapshots))
else:
    maxT = len(chronology.snapshots)
behaviours = chronology.behaviour_labels


# Create save file
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
str_time = time.strftime("%Y%m%d-%H%M%S")
save_name = args.save_name + "_" if args.save_name is not None else ""
save_file = os.path.join(args.save_folder, f"{save_name}{str_time}{'_neighbours' if args.show_neighbours else ''}{'_vision' if args.show_vision_field else ''}_timesteps={maxT}_fps={args.fps}.gif")



# Run visualization
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[-1].axis("off")
ax = axs[0]
ax.axis([minX, maxX, minY, maxY])
cols = {}
for i, label in enumerate(behaviours):
    l, = ax.plot([], [], "o", color=colormaps.tab20(i), label=label)
    cols[label] = l

edges = None
if args.show_neighbours:
    edges = []

    # Add placeholder lines for legend
    red_line, = ax.plot([0,0], [1,1], color='red', label='Close neighbours')
    blue_line, = ax.plot([0,1], [0,1], color='blue', label='Distant neighbours')
    edges.extend([red_line, blue_line])

visions = None
if args.show_vision_field:
    visions = []

ax.legend(bbox_to_anchor=(1.05, 1))


# Create animation
pbar = tqdm.tqdm(total=maxT)
def animate(i):
    # Remove edges
    if edges is not None:
        for edge in edges:
            edge.remove()
        edges.clear()
    # Remove vision fields
    if visions is not None:
        for vision in visions:
            vision.remove()
        visions.clear()

    # Add agents
    snapshot = chronology.snapshots[i] # Get snapshot (direct access to the list, not by timestep)
    states_x = {label : [] for label in behaviours}
    states_y = {label : [] for label in behaviours}
    if snapshot is not None:
        for state in snapshot.states.values():
            x, y, z = state.coordinates
            label = state.behaviour
            states_x[label].append(x)
            states_y[label].append(y)

            # Add edges
            if edges is not None:
                for neighbour_id in state.close_neighbours:
                    neighbour = snapshot.states[neighbour_id]
                    neighbour_x, neighbour_y, _ = neighbour.coordinates
                    edge, = ax.plot([x, neighbour_x], [y, neighbour_y], color="red", alpha=0.2)
                    edges.append(edge)
                for neighbour_id in state.distant_neighbours:
                    neighbour = snapshot.states[neighbour_id]
                    neighbour_x, neighbour_y, _ = neighbour.coordinates
                    edge, = ax.plot([x, neighbour_x], [y, neighbour_y], color="blue", alpha=0.2)
                    edges.append(edge)

            # Add vision field
            if visions is not None and state.past_state is not None:
                past_x, past_y, _ = state.past_state.coordinates
                direction = np.array([x - past_x, y - past_y])
                if np.linalg.norm(direction) > 0:
                    direction /= np.linalg.norm(direction)
                    angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
                    arc = patches.Wedge((x, y), 0.05, angle - 30, angle + 30, color=cols[label].get_color(), alpha=0.3)
                    ax.add_patch(arc)
                    visions.append(arc)
    
    # Print agents
    for label in behaviours:
        cols[label].set_data(states_x[label], states_y[label])

    pbar.update(1)

ani = animation.FuncAnimation(fig, animate, frames=maxT)

# Save animation
writergif = animation.PillowWriter(fps=args.fps)
ani.save(save_file, writer=writergif)
