using PyCall

ipd = pyimport("IPython.display")
play_audio(signal, sample_rate) = ipd.Audio(PyReverseDims(signal), rate=sample_rate)
play_audio(file_name::String) = ipd.Audio(file_name)