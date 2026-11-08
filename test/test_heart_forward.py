"""
Forward modelling and data handling functions of beat.heart on the GF stores
of the synthetic projects (homogeneous full-space, see synthetic.py), with
analytical expectations where possible.
"""

import numpy as num
import pytest
from numpy.testing import assert_allclose
from pyrocko import cake, gf, trace
from pyrocko import util as putil

from beat import heart

km = 1000.0


@pytest.fixture(scope="module")
def engine(seismic_project, geodetic_project):
    return gf.LocalEngine(
        store_superdirs=[
            seismic_project.store_superdir,
            geodetic_project.store_superdir,
        ]
    )


@pytest.fixture(scope="module")
def stores():
    import synthetic

    return SimpleStores(synthetic.seismic_store_id(), synthetic.geodetic_store_id())


class SimpleStores:
    def __init__(self, seismic, geodetic):
        self.seismic = seismic
        self.geodetic = geodetic


def make_targets(store_id, distances, channel="Z", azimuth=0.0):
    targets = []
    for i, dist in enumerate(distances):
        cha_azimuth, dip = {"Z": (0.0, -90.0), "N": (0.0, 0.0), "E": (90.0, 0.0)}[
            channel
        ]
        targets.append(
            heart.DynamicTarget(
                codes=("", "S%i" % i, "", channel),
                lat=10.0,
                lon=20.0,
                north_shift=dist * num.cos(num.deg2rad(azimuth)),
                east_shift=dist * num.sin(num.deg2rad(azimuth)),
                azimuth=cha_azimuth,
                dip=dip,
                store_id=store_id,
                interpolation="multilinear",
            )
        )
    return targets


def make_source(**kwargs):
    d = dict(
        lat=10.0,
        lon=20.0,
        depth=5 * km,
        strike=30.0,
        dip=60.0,
        rake=90.0,
        magnitude=5.0,
        time=putil.str_to_time("2020-01-01 00:00:00"),
        stf=gf.HalfSinusoidSTF(duration=1.0, anchor=-1.0),
    )
    d.update(kwargs)
    return gf.DCSource(**d)


ARRIVAL_TAPER = heart.ArrivalTaper(a=-4.0, b=-2.0, c=8.0, d=10.0)
FILTERER = [heart.Filter(lower_corner=0.05, upper_corner=0.5, order=3)]


class TestArrivalTimes:
    @pytest.mark.parametrize(
        "wavename, velocity", [("any_P", 5500.0), ("any_S", 3200.0)]
    )
    def test_homogeneous_fullspace(self, engine, stores, wavename, velocity):
        source = make_source()
        for target in make_targets(stores.seismic, [10 * km, 30 * km, 50 * km]):
            arrival = heart.get_phase_arrival_time(
                engine, source, target, wavename=wavename, snap=False
            )
            distance = num.sqrt(target.distance_to(source) ** 2 + source.depth**2)
            # straight rays, the travel time tables of the store are
            # interpolated with a tolerance of half a sample
            deltat = 1.0 / engine.get_store(stores.seismic).config.sample_rate
            assert abs(arrival - source.time - distance / velocity) <= 0.5 * deltat

    def test_snap_to_samples(self, engine, stores):
        source = make_source()
        target = make_targets(stores.seismic, [23.3 * km])[0]
        arrival = heart.get_phase_arrival_time(
            engine, source, target, wavename="any_P", snap=True
        )
        deltat = 1.0 / engine.get_store(stores.seismic).config.sample_rate
        offset = (arrival - source.time) / deltat
        assert_allclose(offset, num.round(offset), atol=1e-6)

    def test_unknown_store(self, engine):
        target = make_targets("not_existing_store", [10 * km])[0]
        with pytest.raises(gf.seismosizer.NoSuchStore):
            heart.get_phase_arrival_time(engine, make_source(), target)

    def test_phase_taperer(self, engine, stores):
        source = make_source()
        target = make_targets(stores.seismic, [20 * km])[0]
        arrival = heart.get_phase_arrival_time(engine, source, target, "any_P")
        taperer = heart.get_phase_taperer(
            engine, source, "any_P", target, ARRIVAL_TAPER
        )
        assert_allclose(
            [taperer.a, taperer.b, taperer.c, taperer.d],
            arrival + num.array([-4.0, -2.0, 8.0, 10.0]),
        )


class TestSeismicSynthetics:
    @pytest.fixture(scope="class")
    def setup(self, engine, stores):
        sources = [make_source(), make_source(north_shift=2 * km, magnitude=4.8)]
        targets = make_targets(stores.seismic, [15 * km, 25 * km, 40 * km])
        return sources, targets

    def synthetics(self, engine, sources, targets, outmode, **kwargs):
        kwargs.setdefault("arrival_times", num.array(None))
        return heart.seis_synthetics(
            engine=engine,
            sources=sources,
            targets=targets,
            arrival_taper=ARRIVAL_TAPER,
            wavename="any_P",
            filterer=FILTERER,
            outmode=outmode,
            **kwargs,
        )

    def test_outmodes_consistent(self, engine, setup):
        sources, targets = setup
        array, tmins = self.synthetics(engine, sources, targets, "array")
        traces, ttmins = self.synthetics(engine, sources, targets, "stacked_traces")

        nsamples = ARRIVAL_TAPER.nsamples(2.0)
        assert array.shape == (len(targets), nsamples)
        assert len(traces) == len(targets)
        assert_allclose(tmins, ttmins)
        for row, tr, tmin in zip(array, traces, tmins):
            assert_allclose(tr.ydata, row)
            assert_allclose(tr.tmin, tmin)

        tapered, _ = self.synthetics(engine, sources, targets, "tapered_data")
        assert len(tapered) == len(targets)
        for source_traces, row in zip(tapered, array):
            assert len(source_traces) == len(sources)
            assert_allclose(sum(tr.ydata for tr in source_traces), row, atol=1e-12)

    def test_stacking_is_linear(self, engine, setup):
        sources, targets = setup
        # common time windows, by default they are from the first source
        arrival_times = num.array(
            [
                heart.get_phase_arrival_time(engine, sources[0], t, "any_P")
                for t in targets
            ]
        )
        kwargs = dict(arrival_times=arrival_times)
        stacked, _ = self.synthetics(engine, sources, targets, "array", **kwargs)
        single = [
            self.synthetics(engine, [source], targets, "array", **kwargs)[0]
            for source in sources
        ]
        assert_allclose(stacked, sum(single), rtol=0, atol=1e-12)
        assert num.abs(stacked).max() > 0.0

    def test_data_outmode_unprocessed(self, engine, setup):
        sources, targets = setup
        data, tmins = heart.seis_synthetics(
            engine=engine,
            sources=sources[:1],
            targets=targets,
            arrival_taper=None,
            wavename="any_P",
            filterer=None,
            outmode="data",
            arrival_times=num.array(None),
        )
        response = engine.process(sources=sources[:1], targets=targets)
        for tr, (_, _, ref) in zip(data, response.iter_results()):
            assert_allclose(tr.ydata, ref.ydata)
            assert_allclose(tr.tmin, ref.tmin)

    def test_invalid_outmode(self, engine, setup):
        sources, targets = setup
        with pytest.raises(heart.StackingError):
            self.synthetics(engine, sources, targets, "not_existing")

    def test_taper_filter_traces(self, engine, setup):
        sources, targets = setup
        data, _ = heart.seis_synthetics(
            engine=engine,
            sources=sources[:1],
            targets=targets,
            arrival_taper=None,
            wavename="any_P",
            filterer=None,
            outmode="data",
            arrival_times=num.array(None),
        )
        arrival_times = num.array(
            [
                heart.get_phase_arrival_time(engine, sources[0], t, "any_P")
                for t in targets
            ]
        )
        processed = heart.taper_filter_traces(
            data,
            arrival_taper=ARRIVAL_TAPER,
            filterer=FILTERER,
            arrival_times=arrival_times,
            outmode="array",
            chop_bounds=["b", "c"],
        )
        reference, _ = self.synthetics(engine, sources[:1], targets, "array")
        assert_allclose(
            processed, reference, rtol=0, atol=1e-6 * num.abs(reference).max()
        )

        traces = heart.taper_filter_traces(
            data,
            arrival_taper=ARRIVAL_TAPER,
            filterer=FILTERER,
            arrival_times=arrival_times,
            outmode="stacked_traces",
            chop_bounds=["a", "d"],
        )
        for tr, at in zip(traces, arrival_times):
            assert_allclose(tr.tmin, at + ARRIVAL_TAPER.a, atol=0.5)
            assert tr.ydata.size == ARRIVAL_TAPER.nsamples(2.0, chop_bounds=["a", "d"])


class TestGeodeticSynthetics:
    @pytest.fixture(scope="class")
    def setup(self, stores):
        sources = [
            make_source(time=0.0),
            make_source(time=0.0, east_shift=3 * km, depth=3 * km, rake=0.0),
        ]
        norths, easts = num.meshgrid(num.linspace(-15, 15, 7), num.linspace(-15, 15, 5))
        targets = [
            gf.StaticTarget(
                lats=num.full(norths.size, 10.0),
                lons=num.full(norths.size, 20.0),
                north_shifts=norths.ravel() * km,
                east_shifts=easts.ravel() * km,
                interpolation="multilinear",
                store_id=stores.geodetic,
            )
            for _ in range(2)
        ]
        targets[1].north_shifts = targets[1].north_shifts[:10]
        targets[1].east_shifts = targets[1].east_shifts[:10]
        targets[1].lats = targets[1].lats[:10]
        targets[1].lons = targets[1].lons[:10]
        return sources, targets

    def test_outmodes(self, engine, setup):
        sources, targets = setup
        stacked = heart.geo_synthetics(engine, targets, sources, "stacked_arrays")
        assert [s.shape for s in stacked] == [(35, 3), (10, 3)]

        arrays = heart.geo_synthetics(engine, targets, sources, "arrays")
        assert len(arrays) == len(sources) * len(targets)
        for i_t, target_stack in enumerate(stacked):
            assert_allclose(
                target_stack, arrays[i_t] + arrays[i_t + len(targets)], atol=1e-14
            )

        assert_allclose(
            heart.geo_synthetics(engine, targets, sources, "stacked_array"),
            num.vstack(stacked),
        )
        assert_allclose(
            heart.geo_synthetics(engine, targets, sources, "array"), num.vstack(arrays)
        )
        with pytest.raises(ValueError):
            heart.geo_synthetics(engine, targets, sources, "not_existing")

    def test_north_east_up(self, engine, setup):
        sources, targets = setup
        disp = heart.geo_synthetics(engine, targets[:1], sources[:1], "stacked_arrays")[
            0
        ]
        result = engine.process(sources[:1], targets[:1]).static_results()[0].result
        assert_allclose(disp[:, 0], result["displacement.n"])
        assert_allclose(disp[:, 1], result["displacement.e"])
        assert_allclose(disp[:, 2], -result["displacement.d"])

    def test_static_moment_scaling(self, engine, setup):
        sources, targets = setup
        weak = sources[0].clone(magnitude=sources[0].magnitude - 2.0 / 3.0)
        disp = heart.geo_synthetics(engine, targets[:1], sources[:1], "array")
        disp_weak = heart.geo_synthetics(engine, targets[:1], [weak], "array")
        # moment scales by a factor of 10 for a magnitude difference of 2/3
        assert_allclose(disp, 10.0 * disp_weak, rtol=1e-5, atol=1e-12)


class TestCovariance:
    @pytest.fixture
    def matrices(self):
        rstate = num.random.RandomState(3)
        n = 8
        a = rstate.normal(size=(n, n))
        data = a.dot(a.T) + n * num.eye(n)
        b = rstate.normal(size=(n, n))
        pred_v = 0.1 * b.dot(b.T)
        return data, pred_v

    def test_total_and_log_determinant(self, matrices):
        data, pred_v = matrices
        cov = heart.Covariance(data=data, pred_v=pred_v)
        assert_allclose(cov.c_total, data + pred_v + num.zeros_like(data))
        sign, logdet = num.linalg.slogdet(data + pred_v)
        assert sign > 0
        assert_allclose(cov.log_pdet, logdet, rtol=1e-6)
        assert_allclose(cov.slog_pdet.get_value(), logdet, rtol=1e-6)

    def test_inverse_and_cholesky(self, matrices):
        data, _ = matrices
        cov = heart.Covariance(data=data)
        assert_allclose(cov.inverse(), num.linalg.inv(data), rtol=1e-6, atol=1e-12)
        chol = cov.chol()
        assert_allclose(chol.dot(chol.T), data, rtol=1e-10)
        assert_allclose(num.triu(chol, 1), 0.0)

        weights = cov.chol_inverse
        assert_allclose(weights.T.dot(weights), num.linalg.inv(data), atol=1e-10)

        scaled = cov.inverse(factor=4.0)
        assert_allclose(scaled, num.linalg.inv(4.0 * data), rtol=1e-6, atol=1e-12)

    def test_min_max_components(self, matrices):
        data, pred_v = matrices
        cov = heart.Covariance(data=data, pred_v=pred_v)
        vmin, vmax = cov.get_min_max_components()
        assert vmax >= vmin
        assert vmax == pytest.approx(max(num.abs(data).max(), num.abs(pred_v).max()))


class TestParameter:
    def test_validation(self):
        param = heart.Parameter(
            name="depth",
            lower=num.array([1.0]),
            upper=num.array([5.0]),
            testvalue=num.array([2.0]),
        )
        param.validate_bounds()
        assert param.dimension == 1

        for lower, upper, testvalue in [
            (5.0, 1.0, 2.0),  # lower > upper
            (1.0, 5.0, 6.0),  # testvalue out of bounds
        ]:
            param = heart.Parameter(
                name="depth",
                lower=num.array([lower]),
                upper=num.array([upper]),
                testvalue=num.array([testvalue]),
            )
            with pytest.raises(ValueError):
                param.validate_bounds()

    def test_random_and_repeat(self):
        param = heart.Parameter(
            name="slip",
            lower=num.array([0.5, 1.0]),
            upper=num.array([1.5, 3.0]),
            testvalue=num.array([1.0, 2.0]),
        )
        num.random.seed(0)
        samples = num.array([param.random() for _ in range(500)])
        assert samples.shape == (500, 2)
        assert (samples >= param.lower).all()
        assert (samples <= param.upper).all()
        assert_allclose(samples.mean(0), [1.0, 2.0], atol=0.1)

        assert_allclose(param.get_lower(3), [0.5, 1.0] * 3)
        assert_allclose(param.get_upper(3), [1.5, 3.0] * 3)
        assert_allclose(param.get_testvalue(3), [1.0, 2.0] * 3)


class TestArrivalTaperAndFilters:
    def test_arrival_taper(self):
        assert ARRIVAL_TAPER.duration() == 10.0
        assert ARRIVAL_TAPER.duration(["a", "d"]) == 14.0
        assert ARRIVAL_TAPER.nsamples(2.0) == 20
        assert ARRIVAL_TAPER.nsamples(2.0, chop_bounds=["a", "d"]) == 28
        assert ARRIVAL_TAPER.fadein == 2.0
        assert ARRIVAL_TAPER.fadeout == 2.0

        taper = ARRIVAL_TAPER.get_pyrocko_taper(100.0)
        assert (taper.a, taper.b, taper.c, taper.d) == (96.0, 98.0, 108.0, 110.0)

    def test_sample_rate_consistency(self):
        ARRIVAL_TAPER.check_sample_rate_consistency(0.5)
        with pytest.raises(ValueError):
            heart.ArrivalTaper(
                a=-4.3, b=-2.0, c=8.0, d=10.0
            ).check_sample_rate_consistency(0.5)

    def make_trace(self, frequencies, deltat=0.05, n=4000):
        t = num.arange(n) * deltat
        ydata = sum(num.sin(2 * num.pi * f * t) for f in frequencies)
        return trace.Trace(ydata=ydata, deltat=deltat, tmin=0.0)

    @staticmethod
    def amplitude(tr, frequency):
        spectrum = num.abs(num.fft.rfft(tr.ydata))
        freqs = num.fft.rfftfreq(tr.ydata.size, tr.deltat)
        return spectrum[num.argmin(num.abs(freqs - frequency))]

    def test_bandpass(self):
        tr = self.make_trace([0.02, 0.2, 2.0])
        filtered = tr.copy()
        heart.Filter(lower_corner=0.1, upper_corner=0.5, order=4).apply(filtered)
        for frequency, passes in [(0.02, False), (0.2, True), (2.0, False)]:
            ratio = self.amplitude(filtered, frequency) / self.amplitude(tr, frequency)
            assert (ratio > 0.7) if passes else (ratio < 0.1), frequency

    def test_bandstop(self):
        tr = self.make_trace([0.05, 0.18, 1.0])
        filtered = tr.copy()
        heart.BandstopFilter(lower_corner=0.12, upper_corner=0.25, order=4).apply(
            filtered
        )
        for frequency, passes in [(0.05, True), (0.18, False), (1.0, True)]:
            ratio = self.amplitude(filtered, frequency) / self.amplitude(tr, frequency)
            assert (ratio > 0.7) if passes else (ratio < 0.2), frequency

    def test_filterer_minmax(self):
        filterer = [
            heart.Filter(lower_corner=0.05, upper_corner=0.5),
            heart.Filter(lower_corner=0.1, upper_corner=1.0),
        ]
        fmin, fmax = heart.filterer_minmax(filterer)
        assert fmin == min(filterer[0].get_freqlimits())
        assert fmax == max(filterer[1].get_freqlimits())
        assert fmin < 0.05 and fmax > 1.0


class TestVelocityModels:
    def test_custom_velocity_model(self):
        from synthetic import EVENT, homogeneous_earthmodel

        custom = homogeneous_earthmodel()
        model = heart.get_velocity_model(
            EVENT,
            earth_model_name="ak135-f-continental.m",
            custom_velocity_model=custom,
        )
        assert_allclose(model.material(5 * km).vp, 5500.0)
        assert_allclose(model.material(5 * km).vs, 3200.0)

    def test_ensemble_earthmodel(self):
        reference = cake.load_model("ak135-f-continental.m").extract(depth_max=100 * km)
        num.random.seed(1)
        models = heart.ensemble_earthmodel(
            reference, num_vary=5, error_depth=0.1, error_velocities=0.1
        )
        assert len(models) == 5
        ref_vp = reference.profile("vp")
        for model in models:
            vp = model.profile("vp")
            assert not num.allclose(vp[: ref_vp.size], ref_vp)
            # velocities increase with depth
            assert (num.diff(vp) >= -1e-6).all()
            # Vp/Vs unchanged within layers
            assert_allclose(
                model.profile("vp") / model.profile("vs"),
                reference.profile("vp")[: model.profile("vp").size]
                / reference.profile("vs")[: model.profile("vp").size],
                rtol=0.05,
            )
