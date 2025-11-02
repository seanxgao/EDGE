"""
Project Edge - Minimal Graph-Based Memory MVP
Simple CLI for testing graph memory functionality
"""
import argparse
import sys
from graph.graph_memory import GraphMemory
from core.config import DEFAULT_MEMORY_FILE, DEFAULT_API_KEY_PATH


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Project Edge - Graph Memory MVP")
    parser.add_argument("--add", type=str, help="Add memory text")
    parser.add_argument("--query", type=str, help="Query memories")
    parser.add_argument("--stats", action="store_true", help="Show graph statistics")
    parser.add_argument("--file", type=str, default=None, help="Memory file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose timing logs")
    
    args = parser.parse_args()
    
    # Enable verbose timing if requested
    if args.verbose:
        from core.config import VERBOSE_TIMING
        import core.config
        core.config.VERBOSE_TIMING = True
    
    # Check API key file
    api_key_path = DEFAULT_API_KEY_PATH
    try:
        with open(api_key_path, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
        if not api_key:
            print(f"Error: API key file is empty at {api_key_path}")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: API key file not found at {api_key_path}")
        print("Please ensure the file exists and contains your OpenAI API key")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading API key file: {e}")
        sys.exit(1)
    
    # Initialize graph memory
    try:
        filepath = args.file if args.file else DEFAULT_MEMORY_FILE
        memory = GraphMemory(filepath)
    except Exception as e:
        print(f"Error initializing graph memory: {e}")
        sys.exit(1)
    
    # Handle commands
    if args.add:
        try:
            node_id = memory.add_memory(args.add)
            print(f"Added memory with ID: {node_id}")
            print(f"Text: {args.add}")
        except Exception as e:
            print(f"Error adding memory: {e}")
            sys.exit(1)
    
    elif args.query:
        try:
            results = memory.retrieve_memories(args.query)
            print(f"Query: {args.query}")
            print("Related memories:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result}")
        except Exception as e:
            print(f"Error querying memories: {e}")
            sys.exit(1)
    
    elif args.stats:
        try:
            stats = memory.show_stats()
            print("Graph Statistics:")
            print(f"  Nodes: {stats['nodes']}")
            print(f"  Edges: {stats['edges']}")
            print(f"  Density: {stats['density']}")
            print(f"  File: {stats['filepath']}")
        except Exception as e:
            print(f"Error getting stats: {e}")
            sys.exit(1)
    
    else:
        # Interactive mode
        print("Project Edge - Graph Memory MVP")
        print("Commands: add <text>, query <text>, stats, quit")
        
        while True:
            try:
                cmd = input("\n> ").strip()
                
                if cmd.lower() in ['quit', 'exit', 'q']:
                    break
                elif cmd.lower() == 'stats':
                    stats = memory.show_stats()
                    print(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}, Density: {stats['density']}")
                elif cmd.startswith('add '):
                    text = cmd[4:].strip()
                    if text:
                        node_id = memory.add_memory(text)
                        print(f"Added memory ID: {node_id}")
                    else:
                        print("Please provide text to add")
                elif cmd.startswith('query '):
                    query = cmd[6:].strip()
                    if query:
                        results = memory.retrieve_memories(query)
                        print("Related memories:")
                        for i, result in enumerate(results, 1):
                            print(f"  {i}. {result}")
                    else:
                        print("Please provide query text")
                else:
                    print("Unknown command. Use: add <text>, query <text>, stats, quit")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()