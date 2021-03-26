import pint
ureg = pint.get_application_registry()
# Change global settings of all plots generated in the future

import matplotlib.pyplot as plt
from sugarplot.source.prettify import *
from sugarplot.source.normalization import *
from sugarplot.source.plotters import *
from sugarplot.test.shorthand import *
