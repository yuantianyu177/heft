"""Use geometry-based tracking constants, without annotation-based tuning.

The token stride is 16 pixels for the evaluated models. The localization,
search, and consistency radii are fixed to 1, 3, and 1.5 token strides.
This wrapper leaves all readout, update, and visibility operations unchanged.
"""
import research_track


original_config = research_track.TrackingConfig


def grid_config(**kwargs):
    kwargs.update(argmax_radius=16.0, search_radius=48.0, visibility_threshold=24.0)
    return original_config(**kwargs)


if __name__ == "__main__":
    research_track.TrackingConfig = grid_config
    research_track.main()
