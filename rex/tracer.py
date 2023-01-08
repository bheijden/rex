from typing import Dict, List, Union, Tuple
import sys
from rex.proto import log_pb2


class RecursionDepth:
	def __init__(self, limit):
		self.limit = limit
		self.default_limit = sys.getrecursionlimit()

	def __enter__(self):
		sys.setrecursionlimit(max(self.default_limit, self.limit))

	def __exit__(self, type, value, traceback):
		sys.setrecursionlimit(self.default_limit)


class Step:
	def __init__(self, record: log_pb2.NodeRecord, tick: int):
		"""
		:param record: The node record.
		:param tick: The tick of the step.
		:param static: Whether the step is static or not. This means that the parameters of the step are not changed over time.
		"""
		self._tick = tick
		self._record = record
		self._info = record.info
		self._inputs = {i.info.name: i for i in self._record.inputs}
		self._window = {i.info.output: i.info.window for i in self._record.inputs}
		self._stateful = record.info.stateful
		self._step = record.steps[tick]

		# These variables are set in the .reset() method.
		self._isolate = None
		self._upstream_state: log_pb2.Dependency = None
		self._upstream_inputs: Dict[str, List[log_pb2.Dependency]] = None
		self._steptrace: log_pb2.TracedStep = None

	@property
	def steptrace(self) -> log_pb2.TracedStep:
		"""Makes a copy of the step trace."""
		assert self._steptrace is not None, "You must first trace before you can extract the steptrace."
		steptrace = log_pb2.TracedStep()
		steptrace.CopyFrom(self._steptrace)

		if self._upstream_state is not None:
			steptrace.upstream.append(self._upstream_state)

		# Add upstream input dependencies
		[steptrace.upstream.extend(dependencies) for _, dependencies in self._upstream_inputs.items()]
		return steptrace

	def reset(self, steps: Dict[str, List["Step"]], static: bool, deps: Dict[str, List[bool]] = None, isolate: bool = True):
		"""Reset the step trace.
		:param steps:
		:param static:
		:param deps:
		"""
		# Whether to isolate the step trace in a separate depth or not.
		self._isolate = isolate

		# Create step trace
		self._steptrace = log_pb2.TracedStep(used=False, stateful=self._stateful, static=static, isolate=isolate, name=self._info.name, tick=self._tick, ts_step=self._step.ts_step)

		# Determine if this step will be used (derived from a previous trace).
		is_used = deps[self._info.name][self._tick] if deps is not None else False

		# Determine previous call index that is used
		prev_tick = self._tick - 1
		excluded_ticks = []
		if is_used:
			for idx, prev_is_used in reversed(list(enumerate(deps[self._info.name][:self._tick]))):
				prev_tick = idx
				if prev_is_used:
					break
				else:
					excluded_ticks.append(idx)

		# Does this step have an upstream state dependency
		depends_on_prev_state = prev_tick >= 0
		if (not self._stateful and static and not is_used) or prev_tick in excluded_ticks:
			depends_on_prev_state = False

		# Prepare upstream state dependency
		if depends_on_prev_state:
			prev_source_ts = steps[self._info.name][prev_tick]._step.ts_output
			source = log_pb2.Source(name=self._info.name, tick=prev_tick, ts=prev_source_ts)
			target = log_pb2.Target(name=self._info.name, input_name=None, tick=self._tick, ts=self._step.ts_step)
			self._upstream_state = log_pb2.Dependency(used=False, source=source, target=target)
		else:
			self._upstream_state = None

		# Reroute upstream dependencies from previous excluded steps
		self._upstream_inputs = {}
		for idx in reversed(excluded_ticks):
			excl_step = steps[self._info.name][idx]
			for input_name, i in excl_step._inputs.items():
				self._upstream_inputs[i.info.output] = self._upstream_inputs.get(i.info.output, [])
				for m in i.grouped[excl_step._tick].messages:
					source = log_pb2.Source(name=i.info.output, tick=m.sent.seq, ts=m.sent.ts.sc)
					rerouted = log_pb2.Target(name=self._info.name, input_name=input_name, tick=excl_step._tick, ts=m.received.ts.sc)
					target = log_pb2.Target(name=self._info.name, input_name=input_name, tick=self._tick, ts=m.received.ts.sc, rerouted=rerouted)
					d = log_pb2.Dependency(used=False, source=source, target=target)
					self._upstream_inputs[i.info.output].append(d)

		# Prepare upstream input dependencies
		for input_name, i in self._inputs.items():
			self._upstream_inputs[i.info.output] = self._upstream_inputs.get(i.info.output, [])
			for m in i.grouped[self._tick].messages:
				source = log_pb2.Source(name=i.info.output, tick=m.sent.seq, ts=m.sent.ts.sc)
				# todo: Determine (efficiently) next tick if this message is excluded.
				# Do not iterate over all deps for next uses stepp call,
				# because this will be very inefficient for completely excluded nodes.
				if not is_used and deps is not None:
					rerouted = log_pb2.Target(name="REROUTED", input_name=input_name, tick=self._tick, ts=m.received.ts.sc)
				else:
					rerouted = None
				target = log_pb2.Target(name=self._info.name, input_name=input_name, tick=self._tick, ts=m.received.ts.sc, rerouted=rerouted)
				d = log_pb2.Dependency(used=False, source=source, target=target)
				self._upstream_inputs[i.info.output].append(d)

	def upstream(self, steps: Dict[str, List["Step"]], index: List[int], output: str, num: int) -> int:
		"""Add upstream dependencies to the trace."""
		dependencies = self._upstream_inputs[output]
		depths: List[int] = [0]

		# Determine the number of dependencies to add
		num_add = min(len(dependencies), num)
		num_upstream = max(0, num - len(dependencies))

		# Add input dependencies that were received right before this tick
		assert num_add + num_upstream == num
		for idx, d in enumerate(dependencies[-num_add:]):  # Only add dependencies up to num size.
			if not d.used:  # Only add dependency if it was not previously added (if node is stateless)
				_, depth = steps[output][d.source.tick]._trace(steps, index, d)
				depths.append(depth)

		# Add upstream input dependencies that were received in previous ticks
		prev_tick = self._tick - 1
		if num_upstream > 0 and prev_tick >= 0:
			depth = steps[self._info.name][prev_tick].upstream(steps, index, output, num_upstream)
			depths.append(depth)
		return max(depths)

	def _trace(self, steps: Dict[str, List["Step"]], index: List[int], dependency: log_pb2.Dependency, final: bool = False) -> Tuple[int, int]:
		"""An intermediate trace function that is called recursively."""
		if not self._steptrace.used:
			depths: List[int] = [0]
			# Only trace previous step as dependency if stateful
			if self._upstream_state is not None:
				if not self._upstream_state.used:
					_, depth = steps[self._info.name][self._upstream_state.source.tick]._trace(steps, index, self._upstream_state)
					depths.append(depth)

			# Trace inputs
			# Note: should happen after having added the state dependency, else stateless nodes may have their steps
			# scheduled in non-chronologically order, that in turn, will mess up the circular buffers.
			for output_name, dependencies in self._upstream_inputs.items():
				w = self._window[output_name]
				depth = self.upstream(steps, index, output_name, w)
				depths.append(depth)

			self._steptrace.used = True
			self._steptrace.index = index[0]
			self._steptrace.depth = max(depths) + 1 - int(self._isolate)
			# u = self._steptrace
			# print(f"{u.name=} | {u.tick=} | {u.depth=} ")
			index[0] += 1

		if not final:
			assert not dependency.used, "Cannot add a dependency that was previously added."
			source = dependency.source
			assert source.name == self._info.name, f"Names do not match: {source.name=} vs {self._info.name=}."
			assert source.tick == self._tick, f"Ticks do not match: {source.tick=} vs {self._tick=}."
			assert source.ts == self._step.ts_output, f"Timestamps do not match: {source.ts=} vs {self._step.ts_output=}."
			dependency.used = True
			self._steptrace.downstream.append(dependency)
		return index[0], self._steptrace.depth

	def trace(self, steps: Dict[str, List["Step"]], static: Union[bool, Dict[str, bool]] = False, isolate: bool = True) -> log_pb2.TraceRecord:
		"""Traces the step and returns a TraceRecord."""
		if isinstance(static, bool):
			static = {s: static for s in steps}

		# Prepare steps
		num_steps = sum([len(u) for _, u in steps.items()])

		# Initial trace (with static=True), does not respect chronological order.
		isolate = {name: isolate if name == self._info.name else False for name in steps}
		[[s.reset(steps, static=static.get(s._info.name, False), isolate=isolate.get(s._info.name)) for s in lst] for n, lst in steps.items()]
		with RecursionDepth(num_steps):
			_end, _end_depth = self._trace(steps, [0], log_pb2.Dependency(used=True), final=True)

		# Get dependencies
		deps = {name: [s._steptrace.used for s in lst] for name, lst in steps.items()}

		# Re-trace (with static=Optional[True]), does respect chronological order if we provide deps.
		[[s.reset(steps, static=static.get(s._info.name, False), deps=deps, isolate=isolate.get(s._info.name)) for s in lst] for _, lst in steps.items()]
		with RecursionDepth(num_steps):
			end, end_depth = self._trace(steps, [0], log_pb2.Dependency(used=True), final=True)

		# Gather traceback
		traceback = log_pb2.TraceRecord()

		# Store step from which we started tracing.
		traceback.trace.CopyFrom(self._step)

		# Sort the traced steps
		use: List[log_pb2.TracedStep] = end * [None]
		excluded = []
		pruned = [name for name in steps]
		depths = [[]for i in range(0, 1+_end_depth)]
		for _name, lst in steps.items():
			for step in lst:
				s = step._steptrace
				if s.used:
					# Remove name from pruned list if it was used.
					if s.name in pruned:
						pruned.remove(s.name)

					# Check validity
					for d in s.downstream:
						assert d.used, "Downstream dependency must be used."
					assert s.index < end, "Index violates the bounds"
					assert use[s.index] is None, "Duplicate index"
					# Add to use list at right index
					# Makes a copy below, so changes later on to _steptrace.downstream will not get through.
					use[s.index] = step.steptrace
					depths[s.depth].append(use[s.index])  # NOTE! This connects the steptraces in use and depths
				else:
					assert all([not d.used for d in s.upstream]), "Excluded steptrace must not have used upstream dependencies."
					assert len(s.downstream) == 0, "Removed steps cannot have downstream dependencies"
					excluded.append(step.steptrace)

		# Puts isolated steps in their own depth
		new_depths = []
		for i, depth in enumerate(depths):
			if len(depth) == 0:
				assert i == 0, "Depth must be 0 if empty"
				continue

			# Check if we have isolated steps
			isolated_steps = [u for u in depth if u.isolate]
			other_steps = [u for u in depth if not u.isolate]

			# First add the other steps
			if len(other_steps) > 0:
				new_depths.append(other_steps)

			# Then add the isolated steps (if any, in chronological order)
			isolated_steps.sort(key=lambda u: u.index)
			for u in isolated_steps:
				new_depths.append([u])

		# Update depth and topological indices
		consecutive = 0
		max_consecutive = 0
		topological_index = 0
		for i, depth in enumerate(new_depths):
			# Update depths (takes the isolated depth offset into account)
			for u in depth:
				u.depth = i
				u.index = topological_index
				topological_index += 1

			# Count max sequential steps without an isolated step
			has_isolated = any([u.isolate for u in depth])
			if has_isolated:
				max_consecutive = max(max_consecutive, consecutive)
				consecutive = 0
			else:
				consecutive += 1

		# Update depths to be new_depths (which takes isolated steps into account)
		depths = new_depths

		# Sort used steps by updated index
		use.sort(key=lambda u: u.index)

		# Check validity (according to depth)
		monotone_ticks = {name: -1 for name in steps}
		for i, depth in enumerate(depths):
			names = [u.name for u in depth if not isolate.get(u.name)]
			assert len(names) == len(set(names)), f"Duplicate names in depth {i}: {names}"
			for u in depth:
				# Verify that all upstream dependencies are evaluated before this step
				for d in u.upstream:
					if d.used:
						_has_run = False
						for depthr in reversed(depths[:i]):
							for ur in depthr:
								if ur.name == d.source.name and ur.tick == d.source.tick:
									_has_run = True
									break
							if _has_run:
								break
						assert _has_run, f"Upstream dependency {d.source.name} {d.source.tick} not found in previous depths."

				# Verify that all downstream dependencies are evaluated after this step
				for d in u.downstream:
					assert d.used, "Downstream dependency must be used."
					_has_run = False
					for depthr in depths[i+1:]:
						for ur in depthr:
							if ur.name == d.target.name and ur.tick == d.target.tick:
								_has_run = True
								break
						if _has_run:
							break
				assert monotone_ticks[u.name] < u.tick, f"Steps of node `{u.name}` are scheduled in non-chronological order."
				monotone_ticks[u.name] = u.tick

		# Check validity (according to topological order
		monotone_ticks = {name: -1 for name in steps}
		for i, u in enumerate(use):
			assert u is not None, f"Missing step in used trace for index {i}"

			# Verify that all upstream dependencies are evaluated before this step
			for d in u.upstream:
				if d.used:
					_has_run = False
					for ur in reversed(use[:i]):
						if ur.name == d.source.name and ur.tick == d.source.tick:
							_has_run = True
							break
					assert _has_run, f"Upstream dependency {d.source.name} {d.source.tick} not evaluated before {u.name} {u.tick}."

			# Verify that all downstream dependencies are evaluated after this step
			for d in u.downstream:
				assert d.used, "Downstream dependency must be used."
				_has_run = False
				for ur in use[i:]:
					if ur.name == d.target.name and ur.tick == d.target.tick:
						_has_run = True
						break
				assert _has_run, f"Downstream dependency {d.target.name} {d.target.tick} not evaluated after {u.name} {u.tick}."

			# Check that ticks of step traces are monotonically increasing per node
			assert monotone_ticks[u.name] < u.tick, f"Steps of node `{u.name}` are scheduled in non-chronological order."
			monotone_ticks[u.name] = u.tick

		# Add to traceback
		traceback.name = self._info.name
		traceback.max_depth = depths[-1][0].depth
		traceback.isolate = any(isolate.values())
		traceback.max_consecutive = max_consecutive
		traceback.pruned.extend(pruned)
		traceback.used.extend(use)
		traceback.excluded.extend(excluded)
		return traceback


def trace(record: log_pb2.EpisodeRecord, name: str, tick: int = -1, static: Union[bool, Dict[str, bool]] = False, verbose: bool = True, isolate: bool = True) -> log_pb2.TraceRecord:
	""" Trace a step in the episode record.
	:param record: The episode record to trace.
	:param name: The name of the node to trace.
	:param tick: The tick of the step to trace.
	:param static: Whether to trace static dependencies. If a dictionary is passed, it is used to specify the static dependencies per node.
	:param verbose: Whether to print the trace.
	:param isolate: Whether to isolate the traced node in a separate depth.
	"""

	# Prepare steps of work
	fn_generate_steps = lambda record: [Step(record, tick=tick) for tick in range(0, len(record.steps))]
	steps: Dict[str, List["Step"]] = {n.info.name: fn_generate_steps(n) for n in record.node}

	# Trace step
	tick = tick % len(steps[name])
	record_trace = steps[name][tick].trace(steps, static=static, isolate=isolate)

	# Add node info
	record_trace.node.extend([n.info for n in record.node])
	record_trace.episode.CopyFrom(record)

	# Analyze traced steps
	if verbose:
		# todo: analyze per node how many steps are used and dependencies are required.
		max_consecutive = record_trace.max_consecutive
		max_depth = max([u.depth for u in record_trace.used])
		num_nodes = len(record.node)
		num_used_nodes = num_nodes - len(record_trace.pruned)
		num_isolated = int(isolate)
		num_other = num_used_nodes - num_isolated
		num_used = len(record_trace.used)
		num_excluded = len(record_trace.excluded)
		num_all = num_used + num_excluded
		deps_used = 0
		deps_excluded = 0
		depths = [[] for _ in range(max_depth + 1)]
		num_depths = len(depths)
		num_isolated = tick + 1 if isolate else 0
		for t in record_trace.used:
			# Count used & excluded dependencies
			for d in t.downstream:
				if d.used:
					deps_used += 1
			for d in t.upstream:
				if not d.used:
					deps_excluded += 1
		for t in record_trace.excluded:
			for d in t.upstream:
				if not d.used:
					deps_excluded += 1
			deps_used += len(t.downstream)
		deps_all = deps_used + deps_excluded
		seq_steps = f"{num_used}/{num_all}"
		seq_deps = f"{deps_used}/{deps_all}"
		batch_steps = f"{num_isolated + (num_depths-num_isolated) * num_other}/{num_all}"
		if isolate:
			vec_steps = f"{num_isolated + (num_isolated-1)*max_consecutive * num_other}/{num_all} ({max_consecutive=})"
		else:
			vec_steps = "N/A"

		# NOTE: for batch and vec, the number of steps means step calls. Not to be confused with the number of environment steps.
		print(f"Trace | {name=} | {tick=} | deps (seq): {seq_deps} | step (seq): {seq_steps} | steps (batch): {batch_steps} | steps (vec): {vec_steps}")

	return record_trace