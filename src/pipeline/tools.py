from langchain_core.tools import tool

@tool
def lookup_chapter(chapter_number: int) -> str:
    """
    Retorna o título principal e os tópicos detalhados de um capítulo 
    específico do livro Pro Git (Capítulos de 1 a 10). Use esta ferramenta 
    quando o usuário perguntar o que é abordado em determinado capítulo 
    ou quiser saber a estrutura macro do livro antes de buscar detalhes.
    """
    book_index = {
        1: "Capítulo 1: Getting Started - Topics: About Version Control, Git History, Installing Git, First-Time Setup",
        2: "Capítulo 2: Git Basics - Topics: Getting a Git Repository, Recording Changes, Viewing History, Undoing Things, Remoting, Tagging",
        3: "Capítulo 3: Git Branching - Topics: Branches in a Nutshell, Basic Branching and Merging, Branch Management, Branching Workflows, Remote Branches, Rebasing",
        4: "Capítulo 4: Git on the Server - Topics: The Protocols, Generating SSH Public Key, Git Daemon, Smart HTTP, GitLab",
        5: "Capítulo 5: Distributed Git - Topics: Distributed Workflows, Contributing to a Project, Maintaining a Project",
        6: "Capítulo 6: GitHub - Topics: Account Setup, Contributing to a Project, Maintaining a Project, Managing Organizations",
        7: "Capítulo 7: Git Tools - Topics: Revision Selection, Interactive Staging, Stashing and Cleaning, Reset Demystified, Advanced Merging, Submodules, Credential Storage",
        8: "Capítulo 8: Customizing Git - Topics: Git Configuration, Git Attributes, Git Hooks, Policy Enforcement Script",
        9: "Capítulo 9: Git and Other Systems - Topics: Git as a Client, Migrating to Git",
        10: "Capítulo 10: Git Internals - Topics: Plumbing and Porcelain, Git Objects, Packfiles, Transfer Protocols, Maintenance and Data Recovery"
    }
    
    return book_index.get(
        chapter_number, 
        "Capítulo inválido. O livro Pro Git possui apenas os capítulos de 1 a 10."
    )

def get_tools():
    """Retorna a lista de ferramentas disponíveis para o modelo de linguagem."""
    return [lookup_chapter]