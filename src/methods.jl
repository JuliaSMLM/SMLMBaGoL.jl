# These are methods that must be implemented for each emitter type
# They are used in the RJMCMC algorithm

function move! end
function build_prior_y end
function log_p_z_given_y end
function gen_emitter! end
function gen_emitters end
function merge_emitters end
function gen_observations end
function localization_distance end



