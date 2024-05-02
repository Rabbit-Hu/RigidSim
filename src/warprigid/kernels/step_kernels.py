import warp as wp


@wp.kernel
def kinematic_step_explicit(
    q: wp.array(dtype=wp.transform),
    qd: wp.array(dtype=wp.spatial_vector),
    m: wp.array(dtype=wp.float32),
    com: wp.array(dtype=wp.vec3),
    inertia: wp.array(dtype=wp.mat33),
    h: wp.float32,
    g: wp.vec3,
):
    """RK4 integration for rigid body kinematics
    """

    i = wp.tid()

    w = wp.spatial_top(qd[i])
    v = wp.spatial_bottom(qd[i])
    quat = wp.transform_get_rotation(q[i])
    pos = wp.transform_get_translation(q[i])
    M = m[i]
    C = com[i]
    I = inertia[i]

    wd = wp.inverse(I) * wp.cross(w, I * w)
    vd = g

    v_new = v + vd * h
    w_new = w + wd * h
    qd[i] = wp.spatial_vector(w_new, v_new)

    quat_new = quat + 0.5 * quat * wp.quat(w_new, 0.0) * h
    quat_new /= wp.length(quat_new)
    pos_new = pos + v_new * h
    q[i] = wp.transform(pos_new, quat_new)
    

