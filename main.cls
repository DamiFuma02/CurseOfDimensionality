\ProvidesClass{main}

\newcommand{\MYID}{undefined}
\newcommand{\APIKEY}{undefined}
\newcommand{\USERNAME}{undefined}


\date{}
\LoadClass{article} % Load the base class

\newif\ifreview
\newif\ifpreprint
\newif\ifnoauthor
\reviewfalse
\preprintfalse
\noauthorfalse

\DeclareOption{review}{\reviewtrue}
\DeclareOption{preprint}{\preprinttrue}
\DeclareOption{no_author}{\noauthortrue}
\ProcessOptions\relax

\ifreview
  \RequirePackage{lineno}
  \linenumbers
  \onecolumn
  \renewcommand{\baselinestretch}{2}
\fi

\ifpreprint
  % If preprint option is specified, set the paper to two columns
  \twocolumn
\fi

\ifnoauthor
  % If no_author option is specified, hide the authors' names
  \renewcommand{\maketitle}{}
\fi


\RequirePackage[nopar]{lipsum}
\RequirePackage{graphicx}
\RequirePackage{titling}
\RequirePackage{authblk}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{abstract} 
\RequirePackage{enumitem}
\RequirePackage{outlines}
\RequirePackage[dvipsnames]{xcolor}

\RequirePackage[top=.8in,left=.8in,right=.8in,bottom=1.3in,headheight=55pt]{geometry}


%Define Title formatting 
\newcommand{\articletitle}[1]{\newcommand{\@articletitle}{#1}}
\newcommand{\correspondingauthor}[1]{\newcommand{\@correspondingauthor}{#1}}
\newcommand{\publicationdate}[1]{\newcommand{\@publicationdate}{#1}}
\newcommand{\email}[1]{\newcommand{\@email}{#1}}

%Define Article Types
% Define commands for each article type and a variable to track if one has been set
\newif\ifarticletypechosen
\articletypechosenfalse


\newcommand{\originalarticle}{%
  \ifarticletypechosen
    \PackageError{article}{Multiple article types selected}{Only one article type can be selected}
  \else
    \gdef\articletype{\textcolor{purple}{ORIGINAL ARTICLE}}%
    \articletypechosentrue
  \fi
}
\newcommand{\reviewarticle}{%
  \ifarticletypechosen
    \PackageError{article}{Multiple article types selected}{Only one article type can be selected}
  \else
    \gdef\articletype{\textcolor{OliveGreen}{REVIEW ARTICLE}}%
    \articletypechosentrue
  \fi
}
\newcommand{\translationalresearcharticle}{%
  \ifarticletypechosen
    \PackageError{article}{Multiple article types selected}{Only one article type can be selected}
  \else
    \gdef\articletype{\textcolor{NavyBlue}{TRANSLATIONAL RESEARCH ARTICLE}}%
    \articletypechosentrue
  \fi
}
\newcommand{\editorial}{%
  \ifarticletypechosen
    \PackageError{article}{Multiple article types selected}{Only one article type can be selected}
  \else
    \gdef\articletype{\textcolor{red}{EDITORIAL}}%
    \articletypechosentrue
  \fi
}
\newcommand{\creativework}{%
  \ifarticletypechosen
    \PackageError{article}{Multiple article types selected}{Only one article type can be selected}
  \else
    \gdef\articletype{\textcolor{orange}{CREATIVE WORK}}%
    \articletypechosentrue
  \fi
}

% Define a command for setting the article title
\newcommand{\setarticletitle}[1]{%
  \title{\large{\articletype} \\~\\ \Large\bfseries{#1}}%
}


% Define the header
\pagestyle{fancy}
\fancyhf{}
\setlength{\footskip}{.6in}
\fancyhead[L]{\large\textbf{\@articletitle}} % Replace with your actual article title
\fancyhead[R]{%
    \large{UdSU}
}
\rfoot{\thepage} 

\fancypagestyle{plain}{
  \fancyhf{} 
  \setlength{\headheight}{60pt}
  \setlength{\footskip}{27.57227pt}
  \fancyhead[l]{
        \Large{Università degli Studi di Udine} 
        \Large\textbf{Report: AI for Datahealth}
        }
  \fancyfoot[l]{
  \begin{minipage}{0.5\textwidth} % Adjust the width as needed
    \textbf{Corresponding Author:} \@correspondingauthor \\
    \textbf{Publication Date:} \@publicationdate \\
    \textbf{Email:} \@email \\
  \end{minipage}
}
  \fancyfoot[R]{\thepage} 
  \renewcommand{\headrulewidth}{1pt}
}

\setlength{\columnsep}{16pt}

\titleformat{\section}[block]
{\normalfont\Large\bfseries}{\thesection\hspace{5pt}\vrule depth 3pt width 1pt}{5pt}{}

\makeatletter
\def\@fnsymbol#1{\ensuremath{\ifcase#1\or 1\or 2\or 3\or 4\or 5\else 6\fi}}
\makeatother

\raggedright

\makeatletter
\renewcommand{\maketitle}{\bgroup\setlength{\parindent}{0pt}
\begin{flushleft}
  \textbf{\@title}
  \\
  \@author
\end{flushleft}\egroup
}
\makeatother