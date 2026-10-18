"""Freeze qw3's text protocol helpers for the DeepSWE comparison (no inference)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def freeze(source, output):
    output.mkdir(parents=True, exist_ok=True)
    names = ['src/anthropic_adapter.cpp', 'src/anthropic_adapter.hpp',
             'src/harness_semantics.cpp', 'src/harness_semantics.hpp',
             'src/tool_call_stream.cpp', 'src/tool_call_stream.hpp',
             'include/qw3/kvmem_mandatory.hpp', 'third_party/json.hpp']
    files = {}
    for name in names:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    server = (source / 'src/qw3_server.cpp').read_text()
    files['src/qw3_server.cpp'] = hashlib.sha256(server.encode()).hexdigest()
    def part(start, end):
        return server[server.index(start):server.index(end, server.index(start))]
    body = '''#include "anthropic_adapter.hpp"
#include "harness_semantics.hpp"
#include "tool_call_stream.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <random>
#include <regex>
#include <sstream>
#include <unordered_set>
namespace qw3 {
using json = nlohmann::json;
std::string gen_id(const char* prefix) { static std::atomic<unsigned long> n{0}; return std::string(prefix)+std::to_string(++n); }
std::string dump_json(const json& value) { return value.dump(-1,' ',false,json::error_handler_t::replace); }
struct ReasoningSplit {std::string reasoning; std::string content;};
std::string render_content(const json&);
std::string trim_ascii_ws(std::string);
'''
    body += part('ReasoningSplit split_reasoning(', 'enum class StreamPart')
    body += part('std::string render_content(const json &content) {', 'using detail::PreparedVisionRequest;')
    body += part('std::string trim_ascii_ws(std::string s) {', 'json tool_call_delta(')
    body += part('using RenderedMessageSpan =', '// Apply stop sequences:')
    body += r'''
json operate(const json& input) {
  if (input.value("op", "render") == "parse") {
    const auto split=split_reasoning("<think>\n"+input.at("text").get<std::string>());
    const json tools=input.value("tools",json::array());
    const auto calls=parse_tool_calls_xml(split.content,&tools);
    std::string text=split.content;
    if (!calls.empty()) {auto p=text.find("<tool_call>"); if(p!=std::string::npos)text.resize(p);}
    return {{"reasoning",split.reasoning},{"content",text},{"tool_calls",calls}};
  }
  json req; std::string error;
  if (!detail::anthropic_request_to_openai(input.at("request"),req,error,false)) throw std::runtime_error(error);
  std::vector<RenderedMessageSpan> spans; size_t control=0;
  const json tools=req.value("tools",json::array());
  auto prompt=render_messages(req.at("messages"),&tools,true,"",&spans,true,false,detail::HarnessKind::ClaudeCode,true,&control);
  auto plan=detail::derive_harness_semantic_plan(detail::HarnessKind::ClaudeCode,prompt,control,spans);
  json rows=json::array(); for (const auto& s:spans) rows.push_back({{"index",s.message_index},{"role",s.role},{"begin",s.segment_begin},{"end",s.segment_end},{"content_begin",s.content_begin},{"content_end",s.content_end}});
  json pins=json::array(); for (const auto& s:plan.spans) pins.push_back({{"begin",s.begin},{"end",s.end},{"reason",int(s.reason)}});
  return {{"prompt",prompt},{"messages",req.at("messages")},{"tools",tools},{"spans",rows},{"control_end",control},{"pins",pins},{"query_index",plan.current_query_message_index ? json(*plan.current_query_message_index) : json(nullptr)}};
}
}
extern "C" char* qw3_protocol(const char* input) {
 std::string out; try {out=qw3::operate(qw3::json::parse(input)).dump();}
 catch(const std::exception& e) {out=qw3::json({{"error",e.what()}}).dump();}
 auto p=new char[out.size()+1];std::memcpy(p,out.c_str(),out.size()+1);return p;
}
extern "C" void qw3_protocol_free(char* value) {delete[] value;}
'''
    (output / 'protocol.cpp').write_text(body)
    command = ['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-pthread',
               '-I'+str(output/'src'), '-I'+str(output/'include'), '-I'+str(output/'third_party'),
               str(output/'protocol.cpp'), *(str(output/n) for n in names if n.endswith('.cpp')),
               '-o', str(output/'protocol.so')]
    subprocess.run(command, check=True)
    (output/'provenance.json').write_text(json.dumps({'source':str(source),'files':files,
        'extraction':'Unmodified rendering, reasoning split, tool parsing, Anthropic conversion and harness semantic helper bodies. Unique response call IDs supplied by bridge.',
        'wrapper_sha256':hashlib.sha256(body.encode()).hexdigest(),'command':command},indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/home/chaidi/qw3'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();freeze(a.source,a.output)
