import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/**
 * Renders assistant output as markdown.
 *
 * react-markdown does not render raw HTML unless rehype-raw is added, so this
 * stays compatible with the app's `script-src 'self'` CSP and cannot be used
 * to inject markup through model output.
 *
 * Tailwind v4 here has no typography plugin, so every element is styled
 * explicitly rather than relying on `prose`.
 */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown-body text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,

          h1: ({ children }) => (
            <h1 className="text-base font-bold text-slate-100 mt-3 mb-1.5 first:mt-0">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-sm font-bold text-slate-100 mt-3 mb-1.5 first:mt-0">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold text-slate-200 mt-2.5 mb-1 first:mt-0">{children}</h3>
          ),

          ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-0.5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-0.5">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,

          strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
          em: ({ children }) => <em className="italic text-slate-300">{children}</em>,

          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer noopener"
              className="text-emerald-400 underline underline-offset-2 hover:text-emerald-300"
            >
              {children}
            </a>
          ),

          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-[#2b3552] pl-3 my-2 text-slate-400">
              {children}
            </blockquote>
          ),

          hr: () => <hr className="border-[#1d243c] my-3" />,

          code: ({ className, children, ...props }) => {
            // Fenced blocks carry a language- class; bare inline code does not.
            const isBlock = Boolean(className?.startsWith('language-'));
            if (isBlock) {
              return (
                <code className="block font-mono text-[11px] leading-relaxed" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <code
                className="font-mono text-[12px] bg-[#0b0f1a] border border-[#1d243c] rounded px-1 py-0.5 text-emerald-300"
                {...props}
              >
                {children}
              </code>
            );
          },

          pre: ({ children }) => (
            <pre className="bg-[#06080d] border border-[#1d243c] rounded-lg p-3 my-2 overflow-x-auto text-slate-200">
              {children}
            </pre>
          ),

          table: ({ children }) => (
            <div className="overflow-x-auto my-2">
              <table className="w-full text-xs border-collapse">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border border-[#1d243c] bg-[#0e1220] px-2 py-1 text-left font-semibold text-slate-200">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-[#1d243c] px-2 py-1 align-top">{children}</td>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

export default Markdown;
