Basic usage:

Select one or more videos using the "Select Videos" button.
Click "Run" to create a funscript file. Generated funscripts will be named and placed so they will automatically load the next time you open the video in a compatible player.


Advanced settings:

This tool works in two basic stages. First, it uses a ton of math to get a plot of how far apart the two bodies are. This part doesn't have much leeway for tuning based on preferences, but has performance settings.

Larger batch sizes will improve performance, but increase memory usage. The default will use about 4GB.

If you intend to use your computer for other tasks while this is running, you may want to reduce the number of threads so it doesn't uses your whole CPU. This defaults to the number of cores in your machine.

POV Mode:

By default, this uses motion analysis to find the point at which the two bodies meet, then figures out how much they are moving towards or away from each-other.
In POV mode, it just sees how much things are moving towards or away from the bottom center point of the frame.
This is useful when there is a stationary camera and only one actor who is mostly moving up and down, but can cause artifacts if that's not the case. When in doubt, try both and see which works better.

After that, the script does a few more stages that have room for tweaking:

Detrending

This removes movements that take longer than the detrend window, allowing room to amplify faster ones that you will feel more intensely.

For videos with very slow strokes, the default may lose some of the depth, but it's a trade-off. Too long, and repositioning will cause strokes to use a narrower range while it re-acclimates. The default is a good medium, but tastes may vary.

Normalization

This amplifies motion so it uses the full range of your toy. Setting this lower will increase the intensity of the script, though you will lose some variation between thrusts. Once again, the default is a happy medium.

Keyframe Reduction

Disabling this will output the raw motion data in the funscript instead of reducing it to a set of actions suitable for a toy.
Tested hardware (The Handy) couldn't keep up with the amount of data, though this hasn't been tested on heavier-duty devices like the SR6. Still, the raw data could be useful for those who prefer to hand-tailor their scripts, or toys that can handle the denser information.
