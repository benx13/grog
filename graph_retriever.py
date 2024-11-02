from neo4j import GraphDatabase
from pprint import pprint

class GraphRetriever:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.query2 = """
        CALL {
        CALL db.index.fulltext.queryRelationships("relationshipIndex", $prompt + "~") YIELD relationship, score
        RETURN relationship, score, "relationship" as type
        }
        WITH collect({relationship: relationship, type: type, score: score}) as results
        UNWIND results as result
        WITH results, result
        ORDER BY result.score DESC
        LIMIT 40
        WITH collect(result) as sortedResults, max(result.score) as maxScore
        UNWIND sortedResults as result
        WITH sortedResults, maxScore, result
        WITH sortedResults, maxScore, avg(result.score) as avgScore
        UNWIND sortedResults as result
        WITH sortedResults, maxScore, avgScore, result
        WITH sortedResults, maxScore, avgScore,
        sqrt(avg((result.score - avgScore)^2)) as stdDev
        WITH sortedResults, maxScore, avgScore, stdDev,
        maxScore - 1*stdDev as clusterThreshold
        UNWIND sortedResults as result
        WITH result, maxScore, clusterThreshold
        WHERE result.score >= clusterThreshold
        WITH collect(DISTINCT startNode(result.relationship)) + collect(DISTINCT endNode(result.relationship)) as allNodes,
        result, maxScore, clusterThreshold
        UNWIND allNodes as node
        OPTIONAL MATCH (node)-[r]-(neighbor)
        WHERE node.labeled <> 'event' AND neighbor.labeled <> 'event'
        AND node.labeled <> 'organization' AND neighbor.labeled <> 'organization' AND NOT neighbor IN allNodes
        WITH DISTINCT result, maxScore, clusterThreshold, allNodes,
        node, collect({
            neighbor: neighbor, 
            relationship: r,
            metadata: CASE WHEN neighbor.metadata IS NOT NULL THEN neighbor.metadata ELSE null END
        }) as expandedConnections
        WITH result, maxScore, clusterThreshold,
        allNodes + [conn IN expandedConnections | conn.neighbor] as allExpandedNodes,
        result.relationship as originalRelationship,
        [conn IN expandedConnections | conn.relationship] as additionalRelationships
        RETURN
        originalRelationship as relationship,
        result.type as type,
        result.score as score,
        maxScore,
        result.score / maxScore as relativeScore,
        clusterThreshold,
        startNode(originalRelationship) as startNode,
        endNode(originalRelationship) as endNode,
        allExpandedNodes as nodes,
        additionalRelationships,
        CASE WHEN startNode(originalRelationship).metadata IS NOT NULL THEN startNode(originalRelationship).metadata ELSE null END as startNodeMetadata,
        CASE WHEN endNode(originalRelationship).metadata IS NOT NULL THEN endNode(originalRelationship).metadata ELSE null END as endNodeMetadata
        ORDER BY result.score DESC
        """
        self.query1 = """
        CALL {
        CALL db.index.fulltext.queryNodes("nodeIndex2", $prompt +'~') YIELD node, score
        RETURN node, score, "node" as type
        }
        WITH collect({node: node, type: type, score: score}) as results
        UNWIND results as result
        WITH results, result
        ORDER BY result.score DESC
        LIMIT 40
        WITH collect(result) as sortedResults, max(result.score) as maxScore

        UNWIND sortedResults as result
        WITH sortedResults, maxScore, result
        WITH sortedResults, maxScore, avg(result.score) as avgScore

        UNWIND sortedResults as result
        WITH sortedResults, maxScore, avgScore, result
        WITH sortedResults, maxScore, avgScore,
            sqrt(avg((result.score - avgScore)^2)) as stdDev

        WITH sortedResults, maxScore, avgScore, stdDev,
            maxScore - 1*stdDev as clusterThreshold

        UNWIND sortedResults as result
        WITH result, maxScore, clusterThreshold
        WHERE result.score >= clusterThreshold

        WITH result, maxScore, clusterThreshold, result.node as resultNode
        MATCH (resultNode)-[r]-(connectedNode)
        With result, maxScore, clusterThreshold, resultNode,
            collect({
            node: connectedNode, 
            relationship: type(r),
            relationshipDescription: r.description,
            metadata: CASE WHEN connectedNode.metadata IS NOT NULL THEN connectedNode.metadata ELSE null END
            }) as connectedNodes

        RETURN
        resultNode as node,
        result.type as type,
        result.score as score,
        maxScore,
        result.score / maxScore as relativeScore,
        clusterThreshold,
        connectedNodes as connections,
        CASE WHEN resultNode.metadata IS NOT NULL THEN resultNode.metadata ELSE null END as metadata
        ORDER BY result.score DESC
        """
    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def safe_get(self, obj, key, default=''):
        """Safely get a value from a dictionary or object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        try:
            return getattr(obj, key, default)
        except:
            return default

    def process_node(self, node):
        """Process a node and return a tuple of its properties."""
        return (
            self.safe_get(node, 'name'),
            self.safe_get(node, 'labeled'),
            self.safe_get(node, 'description'),
            self.safe_get(node, 'metadata')
        )

    def invoke(self, input_prompt):
        try:
            # Run both queries
            parameters = {"prompt": input_prompt}
            results1 = self.run_query(self.query1, parameters=parameters)
            results2 = self.run_query(self.query2, parameters=parameters)

            nodes = set()
            relations = set()
            context = []
            metadata = {}

            def safe_process_node(node):
                processed = self.process_node(node)
                return processed + (None,) * (4 - len(processed))

            # Process results from the first query
            for result in results1:
                node_data = safe_process_node(self.safe_get(result, 'node'))
                nodes.add(node_data[:3])  # Add name, labeled, description to nodes set
                if node_data[3]:  # If metadata exists
                    metadata[node_data[0]] = node_data[3]  # Use node name as key for metadata
                for conn in self.safe_get(result, 'connections', []):
                    conn_node_data = safe_process_node(self.safe_get(conn, 'node'))
                    nodes.add(conn_node_data[:3])
                    if conn_node_data[3]:
                        metadata[conn_node_data[0]] = conn_node_data[3]
                    relations.add(self.safe_get(conn, 'relationshipDescription'))

            # Process results from the second query
            for result in results2:
                start_node = safe_process_node(self.safe_get(result, 'startNode'))
                end_node = safe_process_node(self.safe_get(result, 'endNode'))
                nodes.add(start_node[:3])
                nodes.add(end_node[:3])
                if start_node[3]:
                    metadata[start_node[0]] = start_node[3]
                if end_node[3]:
                    metadata[end_node[0]] = end_node[3]
                relations.add(self.safe_get(self.safe_get(result, 'relationship'), 'description'))
                for node in self.safe_get(result, 'nodes', []):
                    node_data = safe_process_node(node)
                    nodes.add(node_data[:3])
                    if node_data[3]:
                        metadata[node_data[0]] = node_data[3]
                for rel in self.safe_get(result, 'additionalRelationships', []):
                    relations.add(self.safe_get(rel, 'description'))

            # Format the output
            context.append('*** Nodes:')
            for name, labeled, description in nodes:
                if name and labeled:
                    context.append(f"{name}: is a {labeled} of description: {description}")
                    context.append('-----')
            context.append('\n')
            context.append('*** Relationships')
            for relation in relations:
                if relation:
                    context.append(relation)
                    context.append('-----')

        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally, print more detailed error information
            import traceback
            print(traceback.format_exc())

        finally:
            # Close the connection
            if isinstance(self, GraphRetriever):
                self.close()
            else:
                print("Warning: connection is not an instance of GraphRetriever")
            
            return {
                'context': '\n'.join(context),
                'metadata': {i.replace(',', '').replace(' ', '') for i in metadata.values() if i.replace(',', '').replace(' ', '')}  # Using a set comprehension for efficiency
            }
if __name__ == "__main__":
        # Neo4j connection details
    uri = "bolt://localhost:7687"  # Update with your Neo4j URI
    user = "neo4j"  # Update with your username
    password = "strongpassword"  # Update with your password



    # Create a Neo4j connection
    retriever = GraphRetriever(uri, user, password)



    context = retriever.invoke('Louis MOREAU')

    print('##########################################################')
    print('##########################################################')
    print('##########################################################')

    print(context['context'])

    print('##########################################################')
    print('##########################################################')
    print('##########################################################')

    print(context['metadata'])  


    




