import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc



# Setup the plot environment

# ------- my personal graphing style --------------------

plt.style.use('ggplot')

golden_mean = (np.sqrt(5)-1.0)/2.0 # Aesthetic ratio
fig_width = 8. # Width in inches
fig_height = 1.1*fig_width*golden_mean # Height in inches
fig_size = [fig_width,fig_height] # Figure size
params = {'backend': 'pdf',
'legend.fontsize': 14,
'xtick.labelsize' : 14,
'ytick.labelsize' : 14,
'axes.labelsize' : 16,
'figure.figsize': fig_size}
plt.rcParams.update(params)
rc("font", **{"size": 10, "family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
# -------------------------------------------------------
x = np.linspace(0, 15, 25)

plt.plot(x, x, label=r"$\frac{12}{15}$")
plt.legend()
plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")
plt.save
