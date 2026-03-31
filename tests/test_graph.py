from src.agents.graph import build_graph, compile_graph


class TestGraph:
    def test_build_graph(self):
        graph = build_graph()
        assert graph is not None

    def test_compile_graph(self):
        compiled = compile_graph()
        assert compiled is not None

    def test_graph_has_nodes(self):
        graph = build_graph()
        assert "ingest" in graph.nodes
        assert "profile" in graph.nodes
        assert "preprocess" in graph.nodes
        assert "select" in graph.nodes
        assert "cluster" in graph.nodes
        assert "evaluate" in graph.nodes
        assert "interpret" in graph.nodes
        assert "refinement" in graph.nodes
