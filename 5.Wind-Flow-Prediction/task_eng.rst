task
=======

Develop and test a model based on measurement data, which translates wind speeds into power of a wind farm.
Please send us your approach in form of a jupyter-notebook, in which you document your code as well as describe - preferably with plots - how you tackled the problem and what you observed.


material
========

* Attached you can find two csv-files, that include wind speeds and wind farm power values. The fields are separated by ";" and end their rows on "\n". 
The files contain a timestamp as the first column, at which the forecast was produced; 
the second column contains the timestep, for which the forecast was created or the measurement was performed; 
the third column includes the wind speeds in m/s; 
the fourth column holds the relative power of the wind park.

    * measurement.csv: this data can be used to create the model.
    * forecast.csv: the wind speeds of this file should be translated into power values by your model. The results can then be compared with the included power values.

* Please use Python (e.g. also numpy, pandas and matplotlib) to write a program for analysis and modelling.


hints
========

* Plot the data to get a feeling, how the model could look like.
* Include comments into your code.
* The use of common python packages is helpful and of course permitted.


