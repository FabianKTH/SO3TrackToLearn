import numpy as np
import torch

from TrackToLearn.environments.interpolation import (
    interpolate_volume_at_coordinates,
    torch_trilinear_interpolation)
from TrackToLearn.utils.rotation import get_ico_points
from TrackToLearn.utils.utils import normalize_vectors


def get_neighbourhood_from_peaks(coords, peaks_vol, no_peaks, last_dir):
    ldim = peaks_vol.shape[-1]
    assert ldim >= no_peaks * 3

    # extract peaks at locations
    peaks_at_c = torch_trilinear_interpolation(peaks_vol, coords)[...,
                 :no_peaks * 3]
    # consider peaks in both symmetric directions
    peaks_at_c = torch.cat([torch.zeros_like(peaks_at_c[..., :3]),  # center
                            peaks_at_c,  # peaks
                            -peaks_at_c], dim=-1)  # symmetric other peaks

    # flip peaks to look into current step=direction
    firstpeak = peaks_at_c[..., 3:6]  # :3 is center coord
    cos = torch.nn.CosineSimilarity(dim=-1)
    flip = torch.sign(cos(last_dir, firstpeak))
    flip[flip == 0] = 1.
    peaks_at_c *= flip[..., None]

    # reshape to fitting shape
    peaks_at_c = peaks_at_c.reshape(-1, 3)

    return peaks_at_c


def get_sh(
        segments,
        data_volume,
        add_neighborhood_vox,
        neighborhood_directions,
        history,
        device,
        prev_dir=None,
        neighb_from_peaks=False,
        peak_volume=None,
        no_peaks=3) -> np.ndarray:
    """ Get the sh coefficients at the end of streamlines
    :param neighb_from_peaks:
    :param peak_volume:
    :param no_peaks:
    """

    N, H, P = segments.shape  # H, history, always 1 here
    flat_coords = np.reshape(segments, (N * H, P))

    coords = torch.as_tensor(flat_coords).to(device)
    n_coords = coords.shape[0]

    if add_neighborhood_vox:
        # Extend the coords array with the neighborhood coordinates

        if neighb_from_peaks:
            assert peak_volume is not None

            neigh_dir_peaks = get_neighbourhood_from_peaks(
                    coords,
                    peak_volume,
                    no_peaks,
                    torch.tensor(prev_dir,
                                 device=device,
                                 dtype=torch.float32))
            coords = torch.repeat_interleave(
                    coords,
                    2 * no_peaks + 1,
                    axis=0)

            coords[:, :3] += neigh_dir_peaks
        else:
            coords = torch.repeat_interleave(
                    coords,
                    neighborhood_directions.size()[0],
                    axis=0)
            coords[:, :3] += \
                neighborhood_directions.repeat(n_coords, 1)

        # Evaluate signal as if all coords were independent
        partial_signal = torch_trilinear_interpolation(
                data_volume, coords)

        # add neighbourhood direction to state
        if neighb_from_peaks:
            partial_signal = torch.cat([partial_signal, neigh_dir_peaks],
                                       axis=-1)

        # Reshape signal into (n_coords, new_feature_size)
        new_feature_size = partial_signal.size()[-1] * \
                           neighborhood_directions.size()[0]
    else:
        partial_signal = torch_trilinear_interpolation(
                data_volume,
                coords).type(torch.float32)
        new_feature_size = partial_signal.size()[-1]

    signal = torch.reshape(partial_signal, (N, history * new_feature_size))

    assert len(signal.size()) == 2, signal.size()

    return signal


def get_neighborhood_directions(
        radius: float,
        mode: str = 'star'
        ) -> np.ndarray:
    """ Returns predefined neighborhood directions at exactly `radius` length
        For now: Use the 6 main axes as neighbors directions, plus (0,0,0)
        to keep current position

    Parameters
    ----------
    radius : float
        Distance to neighbors
    mode: str
        Sampling sheme of the 3d neighbourhood. options are "star", "ico" [
        ...] TODO

    Returns
    -------
    directions : `numpy.ndarray` with shape (n_directions, 3)

    Notes
    -----
    Coordinates are in voxel-space
    """
    if mode == 'star':
        axes = np.identity(3)
        directions = np.concatenate(([[0., 0., 0.]], axes, -axes)) * radius

    elif mode == 'center':
        return np.array([[0., 0., 0.]])

    elif mode == 'ico':
        directions = np.concatenate(
                ([[0., 0., 0.]], get_ico_points(fname='ico1.npy')))
        directions *= radius

    elif mode == 'ico-dode':
        directions = np.concatenate(
                ([[0., 0., 0.]], get_ico_points(fname='ico_in_dode.npy')))
        directions *= radius

    elif mode == 'grid':
        directions = np.indices((3, 3, 3)).reshape(3, -1).T.astype(float) - 1
        # directions =  (np.indices((5, 5, 5)).reshape(3, -1).T.astype(float)
        # - 2) * .5 # [-1, -.5, 0., .5, 1]^3
        directions *= radius
    else:
        raise ValueError(
            'possible options for mode are <star>, <ico>, <ico-dode>, <grid>')

    return directions


def has_reached_gm(
    streamlines: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.,
    min_nb_steps: int = 10
):
    """ Checks which streamlines have their last coordinates inside a mask and
    are at least longer than a minimum strealine length.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.
    min_length: float
        Minimum streamline length to end

    Returns
    -------
    inside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's can end after reaching GM.
    """
    return np.logical_and(is_inside_mask(
        streamlines, mask, threshold),
        np.full(streamlines.shape[0], streamlines.shape[1] > min_nb_steps))


def is_inside_mask(
    streamlines: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.
):
    """ Checks which streamlines have their last coordinates inside a mask.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.

    Returns
    -------
    inside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's last coordinate is inside the mask
        or not.
    """
    # Get last streamlines coordinates
    return interpolate_volume_at_coordinates(
        mask, streamlines[:, -1, :], mode='constant', order=0) >= threshold


def is_outside_mask(
    streamlines: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.
):
    """ Checks which streamlines have their last coordinates outside a mask.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.

    Returns
    -------
    outside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's last coordinate is outside the
        mask or not.
    """

    # Get last streamlines coordinates
    return interpolate_volume_at_coordinates(
        mask, streamlines[:, -1, :], mode='constant', order=0) < threshold


def is_too_long(streamlines: np.ndarray, max_nb_steps: int):
    """ Checks whether streamlines have exceeded the maximum number of steps

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_nb_steps : int
        Maximum number of steps a streamline can have

    Returns
    -------
    too_long : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too long or not
    """
    return np.full(streamlines.shape[0], streamlines.shape[1] >= max_nb_steps)


def is_too_curvy(streamlines: np.ndarray, max_theta: float):
    """ Checks whether streamlines have exceeded the maximum angle between the
    last 2 steps

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_theta : float
        Maximum angle in degrees that two consecutive segments can have between
        each other.

    Returns
    -------
    too_curvy : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too curvy or not
    """
    max_theta_rad = np.deg2rad(max_theta)  # Internally use radian
    if streamlines.shape[1] < 3:
        # Not enough segments to compute curvature
        return np.zeros(streamlines.shape[0], dtype=np.uint8)

    # Compute vectors for the last and before last streamline segments
    u = normalize_vectors(streamlines[:, -1] - streamlines[:, -2])
    v = normalize_vectors(streamlines[:, -2] - streamlines[:, -3])

    # Compute angles
    angles = np.arccos(np.sum(u * v, axis=1).clip(-1., 1.))

    return angles > max_theta_rad


def winding(nxyz: np.ndarray) -> np.ndarray:
    """ Project lines to best fitting planes. Calculate
    the cummulative signed angle between each segment for each line
    and their previous one

    Adapted from dipy.tracking.metrics.winding to allow multiple
    lines that have the same length

    Parameters
    ------------
    nxyz : np.ndarray of shape (N, M, 3)
        Array representing x,y,z of M points in N tracts.

    Returns
    ---------
    a : np.ndarray
        Total turning angle in degrees for all N tracts.
    """

    directions = np.diff(nxyz, axis=1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    thetas = np.einsum(
        'ijk,ijk->ij', directions[:, :-1], directions[:, 1:]).clip(-1., 1.)
    shape = thetas.shape
    rads = np.arccos(thetas.flatten())
    turns = np.sum(np.reshape(rads, shape), axis=-1)
    return np.rad2deg(turns)

    # # This is causing a major slowdown :(
    # U, s, V = np.linalg.svd(nxyz-np.mean(nxyz, axis=1, keepdims=True), 0)

    # Up = U[:, :, 0:2]
    # # Has to be a better way than iterare over all tracts
    # diags = np.stack([np.diag(sp[0:2]) for sp in s], axis=0)
    # proj = np.einsum('ijk,ilk->ijk', Up, diags)

    # v0 = proj[:, :-1]
    # v1 = proj[:, 1:]
    # v = np.einsum('ijk,ijk->ij', v0, v1) / (
    #     np.linalg.norm(v0, axis=-1, keepdims=True)[..., 0] *
    #     np.linalg.norm(v1, axis=-1, keepdims=True)[..., 0])
    # np.clip(v, -1, 1, out=v)
    # shape = v.shape
    # rads = np.arccos(v.flatten())
    # turns = np.sum(np.reshape(rads, shape), axis=-1)

    # return np.rad2deg(turns)


def is_looping(streamlines: np.ndarray, loop_threshold: float):
    """ Checks whether streamlines are looping

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    looping_threshold: float
        Maximum angle in degrees for the whole streamline

    Returns
    -------
    too_curvy : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too curvy or not
    """

    angles = winding(streamlines)

    return angles > loop_threshold
