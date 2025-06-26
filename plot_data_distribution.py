import numpy as np
import matplotlib.pyplot as plt

x_pairs = np.array([[5,7],
                    [5,8],
                    [5,9],
                    [5,10],
                    [5,12],
                    [6,7],
                    [6,8],
                    [6,9],
                    [6,11],
                    [6,12],
                    [7,7],
                    [7,8],
                    [7,10],
                    [7,11],
                    [7,12],
                    [8,7],
                    [8,9],
                    [8,10],
                    [8,11],
                    [8,12]])

o_pairs = np.array([[4,6],
                    [5,11],
                    [6,10],
                    [7,9],
                    [8,8],
                    [9,13]])

green_box_corners = np.array([[5,7],
                              [5,12],
                              [8,7],
                              [8,12]])

red_box_corners = np.array([[3,5],
                           [3,14],
                           [10,5],
                           [10,14]])

# create a scatter plot with a light red box backtround and light green box bakcground
# add the x_pairs as dark green X dots and the o_pairs as dark red O dots
plt.figure(figsize=(10, 6))
# add the red box corners as a light red rectangle
plt.fill_betweenx([5, 14], 3, 10, color='lightcoral', alpha=0.5, label='unseen region')
# add the green box corners as a light green rectangle
plt.fill_betweenx([7, 12], 5, 8, color='lightgreen', alpha=0.5, label='training region')
# add grid
plt.grid(True)
plt.scatter(x_pairs[:, 0], x_pairs[:, 1], color='darkgreen', label='train points', marker='x', s=150)
plt.scatter(o_pairs[:, 0], o_pairs[:, 1], color='darkred', label='test points', marker='o', s=150)
plt.xlim(3, 10)
plt.ylim(5, 14)
plt.xlabel('Cylinder Height [cm]', fontsize=14)
plt.ylabel('Bowl Diamter [cm]', fontsize=14)
# plt.title('Data Distribution Visualization')
# larger font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
# add a legend
plt.legend()
plt.show()